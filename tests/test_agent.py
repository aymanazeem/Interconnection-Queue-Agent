from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from src.agent import RECURSION_LIMIT, SYSTEM_PROMPT, Conversation, ask, build_agent
from src.config import settings
from src.tools import GROUNDING_FINDINGS, MAX_QUERY_ROWS, query_queue, search_studies


class RoutingFakeChatModel(BaseChatModel):
    """A stand in that routes by keyword on the first turn, then writes a final answer.

    The real model never runs in the suite. This lets routing and tool execution be checked
    offline, a queue style question reaches query_queue and a why style question reaches
    search_studies. The final message names the tool that ran so a test can assert the route.
    """

    @property
    def _llm_type(self) -> str:
        return "routing-fake"

    def bind_tools(self, tools, **kwargs) -> "RoutingFakeChatModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            reply = AIMessage(content=f"final answer from {last.name}")
            return ChatResult(generations=[ChatGeneration(message=reply)])
        question = next(
            (message.content for message in reversed(messages) if isinstance(message, HumanMessage)),
            "",
        )
        if "why" in question.lower() or "upgrade" in question.lower():
            call = {"name": "search_studies", "args": {"query": question}, "id": "call-1"}
        else:
            call = {"name": "query_queue", "args": {"sql": "SELECT count(*) FROM projects"}, "id": "call-1"}
        routed = AIMessage(content="", tool_calls=[call])
        return ChatResult(generations=[ChatGeneration(message=routed)])


class FakeRetriever:
    """Returns a fixed passage list, standing in for the Chroma retriever with no api call."""

    def __init__(self, passages: list[Document]) -> None:
        self.passages = passages
        self.seen_query: str | None = None

    def invoke(self, query: str) -> list[Document]:
        self.seen_query = query
        return self.passages


class ContextEchoFakeChatModel(BaseChatModel):
    """Answers with the human turns it has seen, so context carry over is observable offline.

    A turn that sees both questions proves the checkpointer threaded the earlier one back in.
    """

    @property
    def _llm_type(self) -> str:
        return "context-echo-fake"

    def bind_tools(self, tools, **kwargs) -> "ContextEchoFakeChatModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        human_turns = [message.content for message in messages if isinstance(message, HumanMessage)]
        reply = AIMessage(content=f"seen {len(human_turns)}: {' | '.join(human_turns)}")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def test_build_agent_constructs_with_a_fake_key_and_binds_both_tools(monkeypatch) -> None:
    # a real ChatOpenAI is built with a fake key, offline, this checks wiring without a model call.
    monkeypatch.setattr(settings, "openai_api_key", "fake-key-for-test")
    agent = build_agent()
    node_names = set(agent.get_graph().nodes)
    assert {"model", "tools"} <= node_names


def test_build_agent_applies_the_recursion_limit(monkeypatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "fake-key-for-test")
    agent = build_agent()
    assert agent.config.get("recursion_limit") == RECURSION_LIMIT


def test_a_queue_style_question_routes_to_query_queue(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dry_run", False)
    monkeypatch.setattr(settings, "openai_api_key", "fake-key-for-test")
    monkeypatch.setattr("src.agent.ChatOpenAI", lambda **kwargs: RoutingFakeChatModel())

    seen = {}

    def fake_query_projects(sql: str) -> list[dict]:
        seen["sql"] = sql
        return [{"queue_id": "ac2115", "cost_per_kw": 90.0}]

    def fail_retriever(*args, **kwargs):
        raise AssertionError("a queue style question must not reach search_studies")

    monkeypatch.setattr("src.tools.query_projects", fake_query_projects)
    monkeypatch.setattr("src.tools.get_retriever", fail_retriever)

    answer = ask("which solar projects have the highest cost per kW")
    assert seen["sql"].lower().startswith("select")
    assert "query_queue" in answer
    assert isinstance(answer, str) and answer


def test_a_why_style_question_routes_to_search_studies(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dry_run", False)
    monkeypatch.setattr(settings, "openai_api_key", "fake-key-for-test")
    monkeypatch.setattr("src.agent.ChatOpenAI", lambda **kwargs: RoutingFakeChatModel())

    retriever = FakeRetriever(
        [Document(page_content="thermal overloads drive the upgrade cost.",
                  metadata={"queue_id": "ac2115", "source": "ac2115.pdf"})]
    )
    monkeypatch.setattr("src.tools.get_retriever", lambda *args, **kwargs: retriever)

    def fail_query(*args, **kwargs):
        raise AssertionError("a why style question must not reach query_queue")

    monkeypatch.setattr("src.tools.query_projects", fail_query)

    answer = ask("why are the network upgrade costs so high for this cluster")
    assert retriever.seen_query
    assert "search_studies" in answer
    assert isinstance(answer, str) and answer


def test_dry_run_returns_the_canned_string_and_makes_no_model_call(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dry_run", True)

    def blow_up(*args, **kwargs):
        raise AssertionError("dry run must not construct or call the model")

    monkeypatch.setattr("src.agent.ChatOpenAI", blow_up)

    answer = ask("which projects are most at risk")
    assert "dry run" in answer
    assert "query_queue" in answer and "search_studies" in answer
    assert "which projects are most at risk" in answer


def test_build_agent_accepts_a_checkpointer_and_keeps_the_recursion_limit(monkeypatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "fake-key-for-test")
    agent = build_agent(checkpointer=InMemorySaver())
    assert agent.config.get("recursion_limit") == RECURSION_LIMIT
    assert {"model", "tools"} <= set(agent.get_graph().nodes)


def test_conversation_dry_run_returns_canned_and_builds_no_model(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dry_run", True)

    def blow_up(*args, **kwargs):
        raise AssertionError("dry run must not construct or call the model")

    monkeypatch.setattr("src.agent.ChatOpenAI", blow_up)

    answer = Conversation().ask("which projects are most at risk")
    assert "dry run" in answer
    assert "query_queue" in answer and "search_studies" in answer


def test_conversation_carries_context_across_turns(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dry_run", False)
    monkeypatch.setattr(settings, "openai_api_key", "fake-key-for-test")
    monkeypatch.setattr("src.agent.ChatOpenAI", lambda **kwargs: ContextEchoFakeChatModel())

    conversation = Conversation()
    first = conversation.ask("first question")
    second = conversation.ask("second question")
    # the second turn sees both questions, which only happens if the checkpointer threaded the first.
    assert "first question" in second and "second question" in second
    # the first turn saw only itself, so the history grew rather than resetting each turn.
    assert "first question" in first and "second question" not in first


def test_both_tools_have_names_and_descriptions() -> None:
    assert query_queue.name == "query_queue"
    assert search_studies.name == "search_studies"
    assert query_queue.description.strip()
    assert search_studies.description.strip()


def test_query_queue_description_gives_the_model_the_schema_and_allowed_values() -> None:
    description = query_queue.description
    assert "projects" in description
    for column in ["queue_id", "capacity_mw", "cost_per_kw", "is_withdrawn", "fuel_type", "status"]:
        assert column in description
    for status in ["active", "withdrawn", "operational", "suspended"]:
        assert status in description
    for fuel in ["solar", "wind", "storage", "solar+storage", "gas", "hydro"]:
        assert fuel in description
    # a worked example so the model does not have to guess the SQL shape.
    assert "SELECT" in description


def test_search_studies_description_points_at_narrative_cost_drivers() -> None:
    description = search_studies.description.lower()
    assert "study" in description
    assert "upgrade" in description


def test_query_queue_surfaces_the_read_only_guard_refusal() -> None:
    # DELETE never reaches the database, the select only guard rejects it and the tool returns
    # that failure as text for the model to read.
    result = query_queue.invoke({"sql": "DELETE FROM projects WHERE 1=1"})
    assert "failed" in result.lower()


def test_query_queue_caps_the_rows_it_returns(monkeypatch) -> None:
    many = [{"queue_id": f"id{index}"} for index in range(MAX_QUERY_ROWS * 3)]
    monkeypatch.setattr("src.tools.query_projects", lambda sql: many)
    result = query_queue.invoke({"sql": "SELECT queue_id FROM projects"})
    lines = result.splitlines()
    # capped rows plus one truncation notice line.
    assert len(lines) == MAX_QUERY_ROWS + 1
    assert f"of {len(many)} rows" in result


def test_query_queue_reports_when_no_rows_match(monkeypatch) -> None:
    monkeypatch.setattr("src.tools.query_projects", lambda sql: [])
    result = query_queue.invoke({"sql": "SELECT * FROM projects WHERE false"})
    assert "no rows" in result.lower()


def test_search_studies_tags_each_passage_with_queue_id_and_source(monkeypatch) -> None:
    retriever = FakeRetriever(
        [Document(page_content="thermal overloads require a new transformer.",
                  metadata={"queue_id": "ac2115", "source": "ac2115.pdf"})]
    )
    monkeypatch.setattr("src.tools.get_retriever", lambda *args, **kwargs: retriever)
    result = search_studies.invoke({"query": "why is the cost high"})
    assert "ac2115" in result
    assert "ac2115.pdf" in result
    assert "thermal overloads" in result


def test_search_studies_reports_when_nothing_matches(monkeypatch) -> None:
    monkeypatch.setattr("src.tools.get_retriever", lambda *args, **kwargs: FakeRetriever([]))
    result = search_studies.invoke({"query": "nothing relevant"})
    assert "no study passages" in result.lower()


def test_grounding_findings_are_framed_as_general_research() -> None:
    lowered = GROUNDING_FINDINGS.lower()
    assert "13 percent" in lowered
    assert "75 percent" in lowered
    assert "70 percent" in lowered
    # framed as background so the model does not read them as facts about a queried project.
    assert "background" in lowered or "general" in lowered
    # the headline shares are capacity weighted, stated so the agent does not read them as counts.
    assert "capacity" in lowered


def test_system_prompt_carries_the_findings_and_the_withdrawal_caveats() -> None:
    assert GROUNDING_FINDINGS in SYSTEM_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    assert "query_queue" in lowered and "search_studies" in lowered
    # the two hard rules from the caveats, never certainty and preliminary costs.
    assert "certainty" in lowered
    assert "preliminary" in lowered or "upper bound" in lowered
    # comparing a count based share to a capacity share needs the basis called out.
    assert "counting projects" in lowered or "capacity share" in lowered
