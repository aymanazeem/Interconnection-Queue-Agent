"""The two tools the agent calls, and the background findings its prompt carries.

query_queue runs read only SQL against the structured panel, reusing the select only
guard in ingest_queue so the agent cannot write through it. search_studies retrieves
passages from the embedded study reports and tags each with the project and file it came
from. The withdrawal findings sit here too, as general research the agent module folds
into its system prompt rather than a tool the agent has to call.
"""

import json

import duckdb
from langchain_core.documents import Document
from langchain_core.tools import tool

from src.config import settings
from src.ingest_pdfs import get_retriever
from src.ingest_queue import query_projects

# a careless SELECT could pull thousands of rows into the model context, so cap what the
# tool hands back whatever the query itself asks for.
MAX_QUERY_ROWS = 50

QUERY_QUEUE_DESCRIPTION = """\
Run a read only SQL SELECT against the interconnection queue panel and get the matching rows back.

Use this for anything about which projects exist and their structured facts, capacity, fuel,
status, location, dates, queue age, and cost per kW. It cannot say why a cost is high or what
upgrades a project needs, that lives in the study reports, use search_studies for those.

There is one main table, projects, with these columns:
  queue_id (text, the project identifier, an exact opaque string, do not reformat it),
  project_name (text),
  iso (text),
  capacity_mw (double),
  fuel_type (text),
  status (text),
  request_date (date),
  in_service_date (date),
  county (text),
  state (text),
  poi (text, the point of interconnection),
  cost_per_kw (double, network upgrade cost per kW, null when no study was extracted),
  queue_age_days (integer),
  is_withdrawn (boolean).

status is one of active, withdrawn, operational, suspended, other.
fuel_type is one of solar, wind, storage, solar+storage, gas, hydro, other.

A second table, study_extracts, holds the figures behind cost_per_kw, keyed by queue_id, with
columns studied_mw, poi, commercial_probability, total_network_upgrade_cost_usd,
network_upgrade_share, and notes. total_network_upgrade_cost_usd is the dollar total behind
cost_per_kw. When a question is about a project's cost, select both cost_per_kw and, by joining
study_extracts on queue_id, total_network_upgrade_cost_usd. The study reports do not always state
that dollar total, so the panel is the reliable source for it.

Example, the ten most expensive withdrawn solar projects:
  SELECT queue_id, capacity_mw, cost_per_kw FROM projects
  WHERE fuel_type = 'solar' AND is_withdrawn
  ORDER BY cost_per_kw DESC NULLS LAST LIMIT 10

Only SELECT is allowed. The result is capped at a fixed number of rows, so aggregate or add a
LIMIT when a query could match many projects."""

SEARCH_STUDIES_DESCRIPTION = """\
Search the narrative PJM interconnection study reports and get back the passages most relevant to a query.

This is where the reasons behind the numbers live, network upgrade descriptions, the facilities that
overload, dollar cost estimates in context, lead times, and cost allocation language. Use it whenever
the question is why a cost is high, what upgrades a project needs, or how those costs are shared,
anything the structured panel cannot explain on its own.

Pass a plain language query, not SQL. Each passage comes back tagged with the queue_id and source
file it was drawn from, so cite those when you use it."""

GROUNDING_FINDINGS = """\
General research findings on interconnection withdrawal. Treat these as background knowledge for
interpreting evidence, not as facts about any specific project in the panel.

- Of the generation capacity in megawatts that requested interconnection between 2000 and 2020,
  roughly 13 percent was operating by the end of 2025, and about 75 percent withdrew. These are
  shares of capacity across all fuel types, not counts of projects.
- In PJM, withdrawn projects had far higher mean interconnection costs than the active fleet.
  Cost is the dominant driver of withdrawal.
- Network upgrade costs have averaged around 70 percent of total interconnection costs for
  recently withdrawn projects.
- When a high cost project withdraws, its allocated upgrade costs can shift onto later queued
  projects, which can in turn trigger further withdrawals."""


def _format_rows(rows: list[dict]) -> str:
    """Render query rows as one JSON object per line, capped so a wide result cannot flood context."""
    if not rows:
        return "no rows matched the query."
    shown = rows[:MAX_QUERY_ROWS]
    lines = [json.dumps(row, default=str) for row in shown]
    if len(rows) > MAX_QUERY_ROWS:
        lines.append(f"showing {MAX_QUERY_ROWS} of {len(rows)} rows, refine with aggregation or a LIMIT.")
    return "\n".join(lines)


def _format_passages(passages: list[Document]) -> str:
    """Render retrieved passages with the queue id and source file so the agent can cite them."""
    if not passages:
        return "no study passages matched the query."
    blocks = []
    for passage in passages:
        queue_id = passage.metadata.get("queue_id", "unknown")
        source = passage.metadata.get("source", "unknown")
        blocks.append(f"[queue_id {queue_id}, source {source}]\n{passage.page_content}")
    return "\n\n".join(blocks)


@tool(description=QUERY_QUEUE_DESCRIPTION)
def query_queue(sql: str) -> str:
    """Return rows for a read only SELECT against the panel, capped and formatted as text."""
    try:
        rows = query_projects(sql)
    except (ValueError, duckdb.Error) as error:
        # the model wrote this SQL, so hand the failure back for it to correct and retry.
        return f"query failed, {error}"
    return _format_rows(rows)


@tool(description=SEARCH_STUDIES_DESCRIPTION)
def search_studies(query: str) -> str:
    """Return the study passages most relevant to a query, each tagged with its project and file."""
    passages = get_retriever(k=settings.search_k).invoke(query)
    return _format_passages(passages)
