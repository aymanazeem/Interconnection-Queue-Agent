import shutil
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.config import settings
from src.ingest_pdfs import (
    embed_new_pdfs,
    estimate_embedding_cost,
    existing_queue_ids,
    load_and_chunk_pdf,
    open_store,
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    # redirect the data root so tests never touch the real pdf directory or vector store.
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.pdf_raw_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def fixture_pdf(tmp_path, monkeypatch) -> Path:
    # capture the real fixture path before data_dir is redirected, then redirect and copy in.
    real_pdf = settings.sample_study_pdf
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.pdf_raw_dir.mkdir(parents=True, exist_ok=True)
    # the fixture panel keys this project as ac2115, so the pdf keeps that stem.
    destination = settings.pdf_raw_dir / "ac2115.pdf"
    shutil.copy(real_pdf, destination)
    return destination


def test_load_and_chunk_pdf_tags_every_chunk_with_queue_id_and_source(fixture_pdf) -> None:
    chunks = load_and_chunk_pdf(fixture_pdf)
    assert len(chunks) > 0
    assert all(chunk.metadata["queue_id"] == "ac2115" for chunk in chunks)
    assert all(chunk.metadata["source"] == "ac2115.pdf" for chunk in chunks)
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_estimate_embedding_cost_scales_with_token_count() -> None:
    chunks = [Document(page_content="network upgrade cost estimate " * 50)]
    tokens, cost = estimate_embedding_cost(chunks)
    assert tokens > 0
    assert cost > 0
    assert cost == pytest.approx(tokens / 1_000_000 * 0.02)


def test_dry_run_reports_counts_and_never_constructs_a_real_client(fixture_pdf, monkeypatch) -> None:
    monkeypatch.setattr(settings, "dry_run", True)
    monkeypatch.setattr(settings, "openai_api_key", "fake-key-for-test")

    def _blow_up(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run must not construct a real embeddings client")

    monkeypatch.setattr("src.ingest_pdfs.OpenAIEmbeddings", _blow_up)

    summary = embed_new_pdfs()
    assert summary["pdfs"] == 1
    assert summary["chunks"] > 0
    assert summary["tokens"] > 0


def test_second_run_skips_a_pdf_already_in_the_store(fixture_pdf, monkeypatch) -> None:
    monkeypatch.setattr(settings, "dry_run", False)
    monkeypatch.setattr(
        "src.ingest_pdfs.OpenAIEmbeddings", lambda **kwargs: DeterministicFakeEmbedding(size=16)
    )

    first = embed_new_pdfs()
    assert first["pdfs"] == 1

    second = embed_new_pdfs()
    assert second["pdfs"] == 0


def test_rebuild_re_embeds_a_pdf_already_in_the_store(fixture_pdf, monkeypatch) -> None:
    monkeypatch.setattr(settings, "dry_run", False)
    monkeypatch.setattr(
        "src.ingest_pdfs.OpenAIEmbeddings", lambda **kwargs: DeterministicFakeEmbedding(size=16)
    )

    embed_new_pdfs()
    rebuilt = embed_new_pdfs(rebuild=True)
    assert rebuilt["pdfs"] == 1


def test_existing_queue_ids_reflects_stored_metadata(sandbox) -> None:
    embeddings = DeterministicFakeEmbedding(size=8)
    store = open_store(embeddings)
    store.add_documents([Document(page_content="hello", metadata={"queue_id": "ac2115"})])
    assert existing_queue_ids(store) == {"ac2115"}


def test_store_and_retrieve_round_trip_with_fake_embeddings(sandbox) -> None:
    embeddings = DeterministicFakeEmbedding(size=32)
    store = open_store(embeddings)
    matching = Document(
        page_content="network upgrade cost estimate for the project",
        metadata={"queue_id": "ac2115", "source": "ac2115.pdf"},
    )
    other = Document(
        page_content="unrelated background paragraph about the weather",
        metadata={"queue_id": "z002", "source": "z002.pdf"},
    )
    store.add_documents([matching, other])

    results = store.similarity_search("network upgrade cost estimate for the project", k=1)
    assert results[0].metadata["queue_id"] == "ac2115"
