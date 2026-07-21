"""Central configuration for the whole project.

Every module imports the shared settings instance from here, so model names, paths,
and cost caps live in one place rather than scattered through the feature code. All
paths derive from a single data root, so the data tree can be relocated by changing
one value.
"""

from datetime import date
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root sits one level above this file.
REPO_ROOT = Path(__file__).resolve().parents[1]

# openai list prices in dollars per million tokens, keyed by model so a model swap picks up the
# right price with no change in the calling code. the embedding model bills input only.
MODEL_PRICES: dict[str, dict[str, float]] = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost for the given token counts at a model's list price.

    Raises for a model with no listed price rather than guessing, so an unpriced model is caught
    at the point of use instead of silently reporting a wrong cost.
    """
    price = MODEL_PRICES.get(model)
    if price is None:
        raise KeyError(f"no list price for model {model!r}, add it to MODEL_PRICES in config")
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # empty by default so config loads without a key. modules that spend money check it first.
    openai_api_key: str = ""

    # the env names carry an OPENAI_ prefix, the public attributes stay short.
    chat_model: str = Field(default="gpt-4.1", validation_alias="OPENAI_CHAT_MODEL")
    extract_model: str = Field(default="gpt-4.1-nano", validation_alias="OPENAI_EXTRACT_MODEL")
    embed_model: str = Field(default="text-embedding-3-small", validation_alias="OPENAI_EMBED_MODEL")

    # one ISO keeps the first build tight. the queue panel and the study PDFs both come
    # from PJM, so the structured and document tools line up on the same projects.
    iso: str = "PJM"

    # cost caps. dry_run true means no paid call runs until it is flipped on a small sample.
    max_pdfs: int = 25
    dry_run: bool = True

    # fetch politeness. identifies the script to PJM and spaces out requests as a courtesy.
    pdf_user_agent: str = (
        "interconnection-queue-intelligence-agent/0.1 "
        "(personal research project, contact via github issues)"
    )
    pdf_fetch_delay_seconds: float = 1.0

    # pjm switched to the cluster cycle process on this date. projects from it onward have no
    # individual study PDF at these URLs, so the fetch selection only considers earlier ones.
    pdf_study_cutoff: date = date(2020, 10, 1)

    # chunking for the PDF ingester.
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # extraction pulls a focused set of cost chunks, retrieval_k. the agent's search tool answers
    # open ended questions, so it reads wider to catch cost summary lines, search_k.
    retrieval_k: int = 5
    search_k: int = 8

    # the single root every path below derives from.
    data_dir: Path = REPO_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def queue_raw_dir(self) -> Path:
        return self.raw_dir / "queue"

    @property
    def pdf_raw_dir(self) -> Path:
        return self.raw_dir / "pdfs"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def queue_db_path(self) -> Path:
        return self.processed_dir / "queue.duckdb"

    @property
    def vectors_dir(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def chroma_dir(self) -> Path:
        return self.vectors_dir / "chroma"

    @property
    def fixtures_dir(self) -> Path:
        return self.data_dir / "fixtures"

    @property
    def sample_queue_csv(self) -> Path:
        return self.fixtures_dir / "sample_queue.csv"

    @property
    def sample_study_pdf(self) -> Path:
        return self.fixtures_dir / "sample_study.pdf"

    def model_post_init(self, _context: object) -> None:
        # create the writable output dirs on load so later stages do not have to.
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
