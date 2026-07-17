"""Extract cost fields from PJM study excerpts and write them back into the queue panel.

Each project sends only its top retrieved chunks to the chat model, never the whole
PDF, which is what keeps a nano tier model both cheap and focused on the passages that
actually mention cost. The full extracted record lands in a new study_extracts table,
and cost_per_kw is copied onto the matching row of the existing projects table, since
that is the field the structured query tool already expects to find populated.
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd
import tiktoken
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from src.config import settings
from src.ingest_pdfs import existing_queue_ids, open_store
from src.ingest_queue import LBNL_HEADER_ROW, LBNL_SHEET, query_projects

# a fixed vocabulary query, not a per project question, so retrieval stays cheap and consistent.
RETRIEVAL_QUERY = "network upgrade cost estimate point of interconnection"

# studied capacity and the commercial probability assumption tend to sit in a different
# section of the study than the cost language above, so they need their own query.
CAPACITY_RETRIEVAL_QUERY = "studied injection capacity megawatts commercial probability assumption"

# extra chunks pulled by the capacity query, added on top of the cost query's own k.
CAPACITY_SUPPLEMENT_CHUNKS = 2

# the same vocabulary, used to rank dry run previews without spending on a query embedding.
PREVIEW_KEYWORDS = ("cost", "upgrade", "network", "interconnection", "mw", "capacity", "probability")

EXTRACTION_PROMPT_TEMPLATE = """You are reading excerpts from the PJM interconnection study \
report for queue id {queue_id}.

Extract the fields defined by the response schema using only figures and statements found \
in the excerpts below. Leave a field blank if the excerpts do not state it. Do not estimate \
or infer a number that is not written in the text. Copy every dollar and megawatt figure \
exactly as written, digit for digit.

{context}"""

# gpt-4.1-nano list price, per openai's pricing page, input and output priced differently.
INPUT_PRICE_PER_MILLION_TOKENS = 0.10
OUTPUT_PRICE_PER_MILLION_TOKENS = 0.40

# a small structured record, used only to size the dry run cost estimate.
ESTIMATED_OUTPUT_TOKENS = 120

# used only if the configured extract model is not in tiktoken's built in table.
FALLBACK_ENCODING = "cl100k_base"

# the panel only reserved cost_per_kw, so the rest of the extracted record gets its own
# table, joinable back to the panel on queue_id.
CREATE_STUDY_EXTRACTS_TABLE = """
    CREATE TABLE IF NOT EXISTS study_extracts (
        queue_id TEXT PRIMARY KEY,
        studied_mw DOUBLE,
        poi TEXT,
        commercial_probability DOUBLE,
        total_network_upgrade_cost_usd DOUBLE,
        network_upgrade_share DOUBLE,
        notes TEXT
    )
"""

# the current lbnl snapshot carries no hand extracted cost column, so this only matters
# if a future release of the workbook adds one under this name.
LBNL_COST_COLUMN = "ix_cost_per_kw"


class StudyExtract(BaseModel):
    """A cost record read from a PJM study, every field optional since no study states them all."""

    queue_id: str | None = Field(default=None, description="the project queue id.")
    studied_mw: float | None = Field(
        default=None, description="the injection studied, in megawatts."
    )
    poi: str | None = Field(
        default=None, description="the point of interconnection named in the study."
    )
    commercial_probability: float | None = Field(
        default=None,
        description=(
            "the commercial probability assumption stated in the study, as a fraction between "
            "0 and 1. convert a percentage if that is how the study states it, for example 55 "
            "percent becomes 0.55."
        ),
    )
    total_network_upgrade_cost_usd: float | None = Field(
        default=None, description="the total dollar cost of the required network upgrades."
    )
    network_upgrade_share: float | None = Field(
        default=None,
        description="network upgrades as a share of total interconnection cost, if stated.",
    )
    notes: str | None = Field(
        default=None, description="a one sentence summary of the main cost driver."
    )


def compute_cost_per_kw(extract: StudyExtract) -> float | None:
    """Divide network upgrade cost by studied capacity in kw, or None if either is missing or zero."""
    cost = extract.total_network_upgrade_cost_usd
    studied_mw = extract.studied_mw
    if cost is None or studied_mw is None or studied_mw == 0:
        return None
    return cost / (studied_mw * 1000)


def build_prompt(queue_id: str, chunks: list[Document]) -> str:
    """Build the extraction prompt for one project from its retrieved chunks."""
    context = "\n\n---\n\n".join(chunk.page_content for chunk in chunks)
    return EXTRACTION_PROMPT_TEMPLATE.format(queue_id=queue_id, context=context)


def _cost_vocabulary_hits(text: str) -> int:
    """Count cost vocabulary words in a chunk, used only to rank dry run previews."""
    lowered = text.lower()
    return sum(lowered.count(word) for word in PREVIEW_KEYWORDS)


def preview_chunks(store: Chroma, queue_id: str, k: int) -> list[Document]:
    """Return up to k stored chunks for a project with no query embedding, for dry run previews.

    Real retrieval ranks by similarity to a query, which needs a paid embedding call. This
    ranks by a plain text match against the same cost vocabulary instead, so the preview
    tends to show an actual cost passage rather than a title page or table of contents.
    """
    records = store.get(where={"queue_id": queue_id}, include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(records["documents"], records["metadatas"])
    ]
    documents.sort(key=lambda document: _cost_vocabulary_hits(document.page_content), reverse=True)
    return documents[:k]


def retrieve_chunks_by_vector(
    store: Chroma, query_vector: list[float], queue_id: str, k: int
) -> list[Document]:
    """Return the top k chunks for a project ranked against a precomputed query embedding."""
    return store.similarity_search_by_vector(query_vector, k=k, filter={"queue_id": queue_id})


def retrieve_chunks(
    store: Chroma, cost_vector: list[float], capacity_vector: list[float], queue_id: str, k: int
) -> list[Document]:
    """Return the top k cost chunks for a project, plus a few capacity chunks the cost query misses."""
    primary = retrieve_chunks_by_vector(store, cost_vector, queue_id, k)
    seen = {chunk.page_content for chunk in primary}
    supplement = retrieve_chunks_by_vector(
        store, capacity_vector, queue_id, CAPACITY_SUPPLEMENT_CHUNKS
    )
    extra = [chunk for chunk in supplement if chunk.page_content not in seen]
    return primary + extra


def extract_for_project(chain: Runnable, queue_id: str, prompt: str) -> StudyExtract:
    """Invoke the structured output chain, the known queue id wins over whatever the model guesses."""
    result = chain.invoke(prompt)
    return result.model_copy(update={"queue_id": queue_id})


def already_extracted_queue_ids() -> set[str]:
    """Return queue ids that already have a stored extraction record, empty before the first run."""
    tables = query_projects(
        "SELECT table_name FROM information_schema.tables WHERE table_name = 'study_extracts'"
    )
    if not tables:
        return set()
    rows = query_projects("SELECT queue_id FROM study_extracts")
    return {row["queue_id"] for row in rows}


def write_extract(extract: StudyExtract) -> None:
    """Store the full extraction record and update the panel's cost_per_kw for this project.

    query_projects only allows select statements, so this is the one place that opens its
    own connection, since populating cost_per_kw is the reason this stage exists.
    """
    if extract.queue_id is None:
        raise ValueError("cannot write an extract with no queue_id")
    cost_per_kw = compute_cost_per_kw(extract)
    connection = duckdb.connect(str(settings.queue_db_path))
    try:
        connection.execute(CREATE_STUDY_EXTRACTS_TABLE)
        connection.execute("DELETE FROM study_extracts WHERE queue_id = ?", [extract.queue_id])
        connection.execute(
            "INSERT INTO study_extracts VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                extract.queue_id,
                extract.studied_mw,
                extract.poi,
                extract.commercial_probability,
                extract.total_network_upgrade_cost_usd,
                extract.network_upgrade_share,
                extract.notes,
            ],
        )
        connection.execute(
            "UPDATE projects SET cost_per_kw = ? WHERE queue_id = ?",
            [cost_per_kw, extract.queue_id],
        )
    finally:
        connection.close()


def _extract_encoding() -> tiktoken.Encoding:
    """Return the tiktoken encoding for the configured extraction model."""
    try:
        return tiktoken.encoding_for_model(settings.extract_model)
    except KeyError:
        return tiktoken.get_encoding(FALLBACK_ENCODING)


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Return the dollar cost for the given token counts at the extract model's list price."""
    return (
        input_tokens * INPUT_PRICE_PER_MILLION_TOKENS
        + output_tokens * OUTPUT_PRICE_PER_MILLION_TOKENS
    ) / 1_000_000


def extract_all(limit: int | None = None, rebuild: bool = False) -> dict[str, float]:
    """Extract cost fields for projects already in the vector store, up to the configured cap.

    rebuild reprocesses projects that already have a stored extract, instead of skipping them.
    """
    cap = settings.max_pdfs if limit is None else limit
    store = open_store(embeddings=None)
    candidates = existing_queue_ids(store)
    if not rebuild:
        candidates -= already_extracted_queue_ids()
    pending = sorted(candidates)[:cap]

    if not pending:
        print("no new projects to extract")
        return {"projects": 0, "input_tokens": 0, "cost": 0.0}

    if settings.dry_run:
        return _preview_extraction(store, pending)
    return _run_extraction(store, pending)


def _preview_extraction(store: Chroma, pending: list[str]) -> dict[str, float]:
    """Print the prompt for one sample project and a cost estimate for the full run, no api call."""
    sample_id = pending[0]
    chunks = preview_chunks(store, sample_id, settings.retrieval_k + CAPACITY_SUPPLEMENT_CHUNKS)
    prompt = build_prompt(sample_id, chunks)
    encoding = _extract_encoding()
    tokens_per_project = len(encoding.encode(prompt))
    input_tokens = tokens_per_project * len(pending)
    output_tokens = ESTIMATED_OUTPUT_TOKENS * len(pending)
    cost = _estimate_cost(input_tokens, output_tokens)

    print(f"dry run, sample project {sample_id}")
    print(prompt)
    print(
        f"{len(pending)} projects pending, estimated input tokens {input_tokens}, "
        f"estimated cost ${cost:.4f}"
    )
    return {"projects": len(pending), "input_tokens": input_tokens, "cost": cost}


def _run_extraction(store: Chroma, pending: list[str]) -> dict[str, float]:
    """Extract and write back cost fields for every pending project, then report measured usage."""
    # zero temperature, this is a factual extraction task and should not vary between runs.
    chat = ChatOpenAI(model=settings.extract_model, api_key=settings.openai_api_key, temperature=0)
    chain = chat.with_structured_output(StudyExtract)
    embeddings = OpenAIEmbeddings(model=settings.embed_model, api_key=settings.openai_api_key)
    # both queries are fixed, so embed each once and reuse them for every project.
    cost_vector = embeddings.embed_query(RETRIEVAL_QUERY)
    capacity_vector = embeddings.embed_query(CAPACITY_RETRIEVAL_QUERY)
    encoding = _extract_encoding()

    input_tokens = 0
    output_tokens = 0
    for queue_id in pending:
        chunks = retrieve_chunks(store, cost_vector, capacity_vector, queue_id, settings.retrieval_k)
        prompt = build_prompt(queue_id, chunks)
        extract = extract_for_project(chain, queue_id, prompt)
        write_extract(extract)
        input_tokens += len(encoding.encode(prompt))
        output_tokens += len(encoding.encode(extract.model_dump_json()))
        print(f"{queue_id}: extracted, cost_per_kw {compute_cost_per_kw(extract)}")

    cost = _estimate_cost(input_tokens, output_tokens)
    print(
        f"extracted {len(pending)} projects, measured input tokens {input_tokens}, "
        f"output tokens {output_tokens}, cost ${cost:.4f}"
    )
    return {"projects": len(pending), "input_tokens": input_tokens, "cost": cost}


def find_lbnl_workbook() -> Path | None:
    """The first LBNL workbook in the queue raw directory, or None if there is none."""
    matches = sorted(settings.queue_raw_dir.glob("*.xlsx"))
    return matches[0] if matches else None


def load_lbnl_reference(path: Path) -> dict[str, float] | None:
    """Map queue id to LBNL's hand extracted cost per kw. None if the workbook has no such column."""
    raw = pd.read_excel(path, sheet_name=LBNL_SHEET, header=LBNL_HEADER_ROW, engine="openpyxl")
    if LBNL_COST_COLUMN not in raw.columns:
        return None
    reference = raw[["q_id", LBNL_COST_COLUMN]].dropna()
    return dict(zip(reference["q_id"], reference[LBNL_COST_COLUMN]))


def run_validation() -> None:
    """Print extracted cost_per_kw next to LBNL's reference figure for every project where both exist."""
    workbook = find_lbnl_workbook()
    if workbook is None:
        print("no LBNL workbook in data/raw/queue, skipping validation")
        return
    reference = load_lbnl_reference(workbook)
    if reference is None:
        print("LBNL workbook has no hand extracted cost column, skipping validation")
        return

    extracted = query_projects(
        "SELECT queue_id, cost_per_kw FROM projects WHERE cost_per_kw IS NOT NULL"
    )
    rows = [
        (row["queue_id"], row["cost_per_kw"], reference[row["queue_id"]])
        for row in extracted
        if row["queue_id"] in reference
    ]
    if not rows:
        print("no overlap between extracted projects and the LBNL reference")
        return

    print(f"{'queue_id':<12}{'extracted':>14}{'reference':>14}{'difference':>14}")
    for queue_id, extracted_value, reference_value in rows:
        print(
            f"{queue_id:<12}{extracted_value:>14.2f}{reference_value:>14.2f}"
            f"{extracted_value - reference_value:>14.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cost fields from PJM study PDFs already in the vector store."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="compare extracted cost_per_kw against LBNL reference figures",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="reprocess projects that already have a stored extract"
    )
    args = parser.parse_args()
    if args.validate:
        run_validation()
        return
    extract_all(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
