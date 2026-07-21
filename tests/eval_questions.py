"""A small evaluation of the agent against a handful of realistic questions.

Run this by hand, it is not part of the automatic suite because it calls the api. Each question
records what a good answer should contain, checked with a plain assertion where that is honest
and left for the reader to judge where the question is open ended. Ground truth for the checks
comes from the panel itself, so the eval tracks whatever has been extracted. Set DRY_RUN to
false and provide a key, then run python -m tests.eval_questions. In dry run it walks the same
questions against the canned answers for free, which confirms the wiring without spending.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from src.agent import ask, build_agent
from src.config import settings, token_cost
from src.ingest_pdfs import get_querying_store
from src.ingest_queue import query_projects

# a check reads the final answer and the tools that ran, and returns whether it held.
Check = Callable[[str, list[str]], bool]


@dataclass
class EvalQuestion:
    prompt: str
    checks: list[tuple[str, Check]]
    note: str = ""


def _mentions(text: str) -> Check:
    return lambda answer, tools: text.lower() in answer.lower()


def _states_number(value: int) -> Check:
    # models write large counts with thousands separators, so strip them before matching.
    return lambda answer, tools: str(value) in answer.replace(",", "")


def _cites_total_near(target_usd: float, tolerance_fraction: float = 0.08) -> Check:
    """The answer states a network upgrade total near the panel figure, not a small component line.

    This is the check with teeth. It reads dollar amounts from the answer, written either in millions
    or as full grouped figures, and passes only when one lands near the panel total. Quoting a single
    facility allocation as the total, which the model is prone to do, fails it.
    """
    tolerance = max(target_usd * tolerance_fraction, 5_000_000)

    def check(answer: str, tools: list[str]) -> bool:
        text = answer.lower()
        candidates = [float(m.replace(",", "")) * 1_000_000 for m in re.findall(r"(\d[\d,.]*)\s*million", text)]
        candidates += [float(m.replace(",", "")) for m in re.findall(r"(\d{1,3}(?:,\d{3})+)", text)]
        return any(abs(value - target_usd) <= tolerance for value in candidates)

    return check


def _no_wrong_total(target_usd: float, tolerance_fraction: float = 0.15) -> Check:
    """Passes unless the answer names a network upgrade total that is far from the panel figure.

    Stating the cost per kW and describing individual facilities is fine, so this does not demand
    the dollar total. What it catches is the real failure, a single component figure presented as
    the project's total network upgrade cost.
    """
    tolerance = max(target_usd * tolerance_fraction, 10_000_000)
    # a total network upgrade claim, followed within a short window by the figure it names.
    context = re.compile(r"total[^.\n]{0,60}?network upgrade", re.I)
    # a money figure, dollar prefixed, scaled by a word, or comma grouped, so a bare id digit is ignored.
    money = re.compile(
        r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(million|billion)?"
        r"|(\d[\d,]*(?:\.\d+)?)\s*(million|billion)"
        r"|(\d{1,3}(?:,\d{3})+)",
        re.I,
    )
    scale = {"million": 1_000_000, "billion": 1_000_000_000}

    def _figure(match: re.Match) -> float:
        if match.group(1) is not None:
            return float(match.group(1).replace(",", "")) * scale.get((match.group(2) or "").lower(), 1)
        if match.group(3) is not None:
            return float(match.group(3).replace(",", "")) * scale[match.group(4).lower()]
        return float(match.group(5).replace(",", ""))

    def check(answer: str, tools: list[str]) -> bool:
        for claim in context.finditer(answer):
            figure = money.search(answer[claim.end():claim.end() + 60])
            if figure is None:
                continue
            value = _figure(figure)
            if value >= 1_000_000 and abs(value - target_usd) > tolerance:
                return False
        return True

    return check


def _used_tool(name: str) -> Check:
    return lambda answer, tools: name in tools


def _did_not_use_tool(name: str) -> Check:
    return lambda answer, tools: name not in tools


def _avoids_certainty() -> Check:
    # withdrawal is a risk, not a settled outcome, so a good answer never states it as certain.
    phrases = ("will withdraw", "certain to withdraw", "guaranteed to withdraw", "definitely withdraw")
    return lambda answer, tools: not any(phrase in answer.lower() for phrase in phrases)


def _admits_no_evidence() -> Check:
    phrases = ("no wind", "none", "no known", "not have", "no extracted", "cannot", "no such", "no data")
    return lambda answer, tools: any(phrase in answer.lower() for phrase in phrases)


def _lists_in_order(queue_ids: list[str]) -> Check:
    """Every id appears and their first mentions run in the given order."""
    def check(answer: str, tools: list[str]) -> bool:
        lowered = answer.lower()
        positions = [lowered.find(queue_id.lower()) for queue_id in queue_ids]
        if any(position < 0 for position in positions):
            return False
        return positions == sorted(positions)

    return check


def _withdrawn_solar() -> tuple[int, float]:
    row = query_projects(
        "SELECT count(*) AS n, coalesce(sum(capacity_mw), 0) AS mw "
        "FROM projects WHERE fuel_type = 'solar' AND is_withdrawn"
    )[0]
    return row["n"], row["mw"]


def _costed_ranking() -> list[str]:
    rows = query_projects(
        "SELECT queue_id FROM projects WHERE cost_per_kw > 0 ORDER BY cost_per_kw DESC, queue_id"
    )
    return [row["queue_id"] for row in rows]


def _study_total_usd(queue_id: str) -> float | None:
    """The extracted network upgrade total for one project, used as the panel truth for grounding."""
    rows = query_projects(
        "SELECT total_network_upgrade_cost_usd AS total FROM study_extracts "
        f"WHERE queue_id = '{queue_id}'"
    )
    return rows[0]["total"] if rows and rows[0]["total"] is not None else None


def build_questions() -> list[EvalQuestion]:
    """Assemble the question set, wiring each check to the current panel figures."""
    solar_count, solar_mw = _withdrawn_solar()
    ranking = _costed_ranking()
    top = ranking[0] if ranking else ""
    af1_total = _study_total_usd("AF1-236")
    top_total = _study_total_usd(top) if top else None

    document_checks = [
        ("routes to search_studies", _used_tool("search_studies")),
        ("names the project", _mentions("AF1-236")),
        ("speaks to upgrades", _mentions("upgrade")),
    ]
    if af1_total is not None:
        document_checks.append(("cites the upgrade total, not a component", _cites_total_near(af1_total)))

    combined_checks = [
        ("reads the panel", _used_tool("query_queue")),
        ("reads the studies", _used_tool("search_studies")),
        ("names the top project", _mentions(top)),
    ]
    if top_total is not None:
        combined_checks.append(
            ("does not pass a component off as the total", _no_wrong_total(top_total))
        )

    return [
        EvalQuestion(
            prompt=(
                "How many withdrawn solar projects are in the panel, and what is their combined "
                "capacity in MW?"
            ),
            checks=[
                ("routes to query_queue", _used_tool("query_queue")),
                ("stays out of the study reports", _did_not_use_tool("search_studies")),
                ("states the count", _states_number(solar_count)),
            ],
            note=f"panel truth, {solar_count} withdrawn solar projects, about {solar_mw:,.0f} MW combined.",
        ),
        EvalQuestion(
            prompt=(
                "According to its PJM study report, what network upgrades does queue ID AF1-236 "
                "need, and what does the cost summary give as the total network upgrade cost?"
            ),
            checks=document_checks,
            note="expect the Mackeys 230 kV connection and the roughly 473 million dollar upgrade allocation.",
        ),
        EvalQuestion(
            prompt=(
                "Among projects with a known network upgrade cost per kW, which is the most "
                "expensive, and what upgrades drive that cost?"
            ),
            checks=combined_checks,
            note=f"panel truth, most expensive costed project is {top or 'none yet'}.",
        ),
        EvalQuestion(
            prompt=(
                "Which solar projects look most at risk of withdrawing based on interconnection "
                "cost, and why? Do not overstate certainty."
            ),
            checks=[
                ("reads the panel", _used_tool("query_queue")),
                ("frames risk without certainty", _avoids_certainty()),
            ],
            note="expect high cost per kW solar read against the general findings, framed as risk. eyeball the reasoning.",
        ),
        EvalQuestion(
            prompt=(
                "Which wind projects in the panel have a known network upgrade cost per kW above "
                "300, and what drives it?"
            ),
            checks=[
                ("reads the panel", _used_tool("query_queue")),
                ("admits there is no such evidence", _admits_no_evidence()),
            ],
            note="no wind project has an extracted cost, so a good answer says so rather than inventing one.",
        ),
        EvalQuestion(
            prompt=(
                "Rank the projects that have a known network upgrade cost per kW from highest to "
                "lowest, with their cost per kW."
            ),
            checks=[
                ("reads the panel", _used_tool("query_queue")),
                ("lists them in cost order", _lists_in_order(ranking)),
            ],
            note=f"panel truth, cost order is {ranking or 'none yet'}.",
        ),
    ]


def _tools_called(messages: list[BaseMessage]) -> list[str]:
    return [message.name for message in messages if isinstance(message, ToolMessage)]


def _token_usage(messages: list[BaseMessage]) -> tuple[int, int]:
    """Sum input and output tokens across the run from each reply's usage metadata."""
    input_tokens = 0
    output_tokens = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
    return input_tokens, output_tokens


def _cost(input_tokens: int, output_tokens: int) -> float:
    return token_cost(settings.chat_model, input_tokens, output_tokens)


def answer_question(agent: CompiledStateGraph, prompt: str) -> tuple[str, list[str], int, int]:
    """Run one question and return its answer, the tools it called, and its token counts."""
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    messages = result["messages"]
    input_tokens, output_tokens = _token_usage(messages)
    return messages[-1].text, _tools_called(messages), input_tokens, output_tokens


def run_wiring_pass(questions: list[EvalQuestion]) -> None:
    """Walk the questions against the canned answers, no api call, so the loop can be checked free."""
    print("dry run, no api call. this walks the loop with canned answers.")
    print("set DRY_RUN=false and provide a key for the real eval.\n")
    for index, question in enumerate(questions, start=1):
        print(f"[{index}] {question.prompt}")
        print(f"    {ask(question.prompt)}\n")
    print(f"{len(questions)} questions wired. checks and cost print only in a real run.")


def run_eval(questions: list[EvalQuestion]) -> None:
    """Ask each question for real, run its checks, and report token use and cost as they add up."""
    agent = build_agent()
    # build the vector client on the main thread, the agent calls search_studies from worker threads
    # where creating the native chroma client fresh has proven flaky.
    get_querying_store()
    total_input = 0
    total_output = 0
    passed = 0
    total_checks = 0
    for index, question in enumerate(questions, start=1):
        answer, tools, input_tokens, output_tokens = answer_question(agent, question.prompt)
        total_input += input_tokens
        total_output += output_tokens

        print(f"[{index}] {question.prompt}")
        print(f"    tools called, {', '.join(tools) or 'none'}")
        if question.note:
            print(f"    note, {question.note}")
        print(f"    answer, {answer}")
        for label, check in question.checks:
            ok = check(answer, tools)
            passed += int(ok)
            total_checks += 1
            print(f"    [{'pass' if ok else 'FAIL'}] {label}")
        print(
            f"    tokens this question in {input_tokens} out {output_tokens}, "
            f"running in {total_input} out {total_output}, cost so far ${_cost(total_input, total_output):.4f}\n"
        )

    print(f"checks passed {passed} of {total_checks}")
    print(
        f"measured tokens in {total_input} out {total_output}, "
        f"total cost ${_cost(total_input, total_output):.4f}"
    )


def main() -> None:
    questions = build_questions()
    if settings.dry_run:
        run_wiring_pass(questions)
        return
    run_eval(questions)


if __name__ == "__main__":
    main()
