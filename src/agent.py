"""Question answering agent over the queue panel and the study reports.

Built on LangChain's create_agent rather than a hand written StateGraph. The task is a
plain tool calling loop, reason, pick a tool, read the result, then answer, and the
prebuilt factory carries the message state and tool routing without a custom graph.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from src.config import settings
from src.tools import GROUNDING_FINDINGS, query_queue, search_studies

# a single question must never trigger an unbounded chain of tool calls, this caps the
# worst case cost. ten steps is generous for a reason, call a tool, read it, answer loop.
RECURSION_LIMIT = 10

# a chat session runs on one checkpointer thread, so every turn keys onto the same history.
CHAT_THREAD_ID = "cli-chat"

# the tools the agent may call, listed once so build_agent and the dry run reply stay in step.
AGENT_TOOLS = [query_queue, search_studies]

SYSTEM_PROMPT = f"""You analyze United States electricity interconnection queue projects and their \
engineering study reports. Answer questions by gathering evidence with your tools and reasoning over \
it, never from memory alone.

You have two tools, and they play different roles.
- query_queue runs SQL against the structured panel. It is the source of truth for numbers, which \
projects exist and their capacity, fuel, status, location, dates, and the network upgrade cost. The \
total network upgrade cost and the cost per kW live here, in cost_per_kw and in \
study_extracts.total_network_upgrade_cost_usd. Take any total or cost per kW from this tool.
- search_studies retrieves passages from the narrative study reports. Use it for why a cost is high, \
which facilities must be rebuilt, lead times, and how costs are allocated. Its passages are per \
facility line items and allocation tables, not the project total.
Most worthwhile questions need both. Get the projects and their totals from query_queue, then explain \
the drivers with search_studies.

Numbers rule. The total network upgrade cost and the cost per kW come from the panel, from cost_per_kw \
and study_extracts.total_network_upgrade_cost_usd. Whenever an answer needs a project's total or cost \
per kW, get it from query_queue and state it in dollars, even when the question is phrased around the \
study report, since the panel holds the extracted cost summary total. Do not conclude the total is \
unavailable because the study passages show only component lines. A dollar figure in a study passage is \
almost always one facility or one allocation line, so treat it as a single component, never as the \
project's total, and never label it as panel data. When a study figure and the panel total disagree, \
trust the panel. Do not state a dollar figure that is not in your evidence.

Reasoning about withdrawal risk. Weigh a project's own numbers, a high cost per kW, network upgrades \
as a large share of cost, and cost allocation cascades, against the general findings below. Frame the \
answer as risk factors and the evidence for them. Never state that a project will withdraw as a \
certainty. High cost signals elevated risk, it does not settle the outcome.

{GROUNDING_FINDINGS}

Answering. Cite specific queue IDs and specific study passages. Be explicit about where each number \
comes from, a figure read from a study, a value from the structured panel, or a general research \
finding. When you set a panel figure beside those findings, say whether you are counting projects or \
summing capacity and whether the subset lines up, so a project count is not read as the same basis as \
a capacity share. Quoted study costs are preliminary estimates, not settled amounts, and a withdrawn project \
usually does not pay the full quoted figure, so treat a quoted cost as an upper bound. When the \
evidence is thin, say so rather than filling the gap."""


def build_agent(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Construct the agent with both tools bound, the system prompt set, and the recursion cap applied.

    Pass a checkpointer to keep the running message history across calls, which is how the chat
    loop carries context between questions. The single shot ask path leaves it unset.
    """
    # zero temperature, answers should be reproducible and stay close to the retrieved evidence.
    model = ChatOpenAI(model=settings.chat_model, api_key=settings.openai_api_key, temperature=0)
    agent = create_agent(model, AGENT_TOOLS, system_prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
    agent = agent.with_config({"recursion_limit": RECURSION_LIMIT})
    # confirm the cap took, an unbounded tool loop is the one real cost risk in this component.
    if agent.config.get("recursion_limit") != RECURSION_LIMIT:
        raise RuntimeError("agent recursion limit was not applied")
    return agent


def _dry_run_answer(question: str) -> str:
    """The free stand in for a real answer, names the tools that would run and echoes the question."""
    tool_names = ", ".join(agent_tool.name for agent_tool in AGENT_TOOLS)
    return (
        "dry run, no model call was made. "
        f"the tools that would be available are {tool_names}. "
        f"the question was {question!r}."
    )


def ask(question: str) -> str:
    """Run one question through the agent and return the final answer text.

    In dry run this skips the model and returns a canned reply, so the wiring can be
    exercised with no key and no spend.
    """
    if settings.dry_run:
        return _dry_run_answer(question)
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].text


class Conversation:
    """A chat session that carries context across turns.

    Each turn sends only the new question. A per session checkpointer holds the running
    message history, so the agent sees earlier turns without the caller threading messages by
    hand. In dry run no model is built and every turn returns the canned reply, so the loop
    runs for free.
    """

    def __init__(self) -> None:
        self._dry_run = settings.dry_run
        self._agent = None if self._dry_run else build_agent(checkpointer=InMemorySaver())
        self._config = {"configurable": {"thread_id": CHAT_THREAD_ID}}

    def ask(self, question: str) -> str:
        """Answer one turn, carrying the context of earlier turns in the same session."""
        if self._dry_run:
            return _dry_run_answer(question)
        result = self._agent.invoke({"messages": [HumanMessage(content=question)]}, self._config)
        return result["messages"][-1].text
