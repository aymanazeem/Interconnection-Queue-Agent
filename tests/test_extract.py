import shutil
from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import ValidationError

from src.config import settings
from src.extract import (
    StudyExtract,
    already_extracted_queue_ids,
    build_prompt,
    compute_cost_per_kw,
    extract_all,
    extract_for_project,
    find_lbnl_workbook,
    load_lbnl_reference,
    preview_chunks,
    retrieve_chunks,
    run_validation,
    write_extract,
)
from src.ingest_pdfs import open_store
from src.ingest_queue import (
    LBNL_SHEET,
    build_projects_table,
    normalize_frame,
    query_projects,
    read_fixture,
)


@pytest.fixture
def panel(tmp_path, monkeypatch):
    # mirrors the queue panel fixture pattern, redirect the data root then copy the fixture in.
    real_csv = settings.sample_queue_csv
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.sample_queue_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(real_csv, settings.sample_queue_csv)
    build_projects_table(normalize_frame(read_fixture()), settings.queue_db_path)
    return settings.queue_db_path


@pytest.fixture
def store_with_chunks(panel) -> Chroma:
    # panel already redirected data_dir, so the store persists in the same sandbox.
    embeddings = DeterministicFakeEmbedding(size=16)
    store = open_store(embeddings)
    store.add_documents(
        [
            Document(
                page_content="the study estimates a network upgrade cost of 1500000 dollars for 150 mw.",
                metadata={"queue_id": "ac2115", "source": "ac2115.pdf"},
            ),
            Document(
                page_content="a second passage describing the point of interconnection.",
                metadata={"queue_id": "ac2115", "source": "ac2115.pdf"},
            ),
        ]
    )
    return store


def test_study_extract_validates_a_correct_record() -> None:
    extract = StudyExtract.model_validate(
        {"queue_id": "ac2115", "studied_mw": 150.0, "total_network_upgrade_cost_usd": 4_500_000.0}
    )
    assert extract.queue_id == "ac2115"
    assert extract.studied_mw == 150.0


def test_study_extract_rejects_a_malformed_record() -> None:
    with pytest.raises(ValidationError):
        StudyExtract.model_validate({"studied_mw": "not a number"})


def test_compute_cost_per_kw_divides_cost_by_kw() -> None:
    extract = StudyExtract(total_network_upgrade_cost_usd=1_500_000.0, studied_mw=150.0)
    assert compute_cost_per_kw(extract) == pytest.approx(10.0)


def test_compute_cost_per_kw_is_none_when_cost_is_missing() -> None:
    extract = StudyExtract(studied_mw=150.0)
    assert compute_cost_per_kw(extract) is None


def test_compute_cost_per_kw_is_none_when_studied_mw_is_zero() -> None:
    extract = StudyExtract(total_network_upgrade_cost_usd=1000.0, studied_mw=0.0)
    assert compute_cost_per_kw(extract) is None


def test_build_prompt_includes_the_queue_id_and_chunk_text() -> None:
    chunk = Document(page_content="upgrade cost is five million dollars.", metadata={})
    prompt = build_prompt("ac2115", [chunk])
    assert "ac2115" in prompt
    assert "upgrade cost is five million dollars." in prompt


def test_preview_chunks_favors_the_chunk_that_mentions_cost(store_with_chunks) -> None:
    top = preview_chunks(store_with_chunks, "ac2115", k=1)
    assert "network upgrade cost" in top[0].page_content


def test_retrieve_chunks_adds_the_capacity_supplement_without_duplicating(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    embeddings = DeterministicFakeEmbedding(size=16)
    store = open_store(embeddings)
    cost_chunk = Document(page_content="cost chunk", metadata={"queue_id": "ac2115"})
    capacity_chunk = Document(page_content="capacity chunk", metadata={"queue_id": "ac2115"})
    store.add_documents([cost_chunk, capacity_chunk])

    cost_vector = embeddings.embed_query("cost chunk")
    capacity_vector = embeddings.embed_query("capacity chunk")
    combined = retrieve_chunks(store, cost_vector, capacity_vector, "ac2115", k=1)

    contents = {chunk.page_content for chunk in combined}
    assert contents == {"cost chunk", "capacity chunk"}


def test_extract_for_project_overrides_the_model_returned_queue_id() -> None:
    chain = MagicMock()
    chain.invoke.return_value = StudyExtract(queue_id="wrong-id", studied_mw=100.0)
    result = extract_for_project(chain, "ac2115", "a prompt")
    assert result.queue_id == "ac2115"
    assert result.studied_mw == 100.0


def test_write_extract_updates_the_panel_cost_per_kw(panel) -> None:
    extract = StudyExtract(
        queue_id="ac2115",
        studied_mw=150.0,
        total_network_upgrade_cost_usd=1_500_000.0,
        notes="upgrade driven by thermal overloads.",
    )
    write_extract(extract)

    rows = query_projects("SELECT cost_per_kw FROM projects WHERE queue_id = 'ac2115'")
    assert rows[0]["cost_per_kw"] == pytest.approx(10.0)

    stored = query_projects("SELECT notes FROM study_extracts WHERE queue_id = 'ac2115'")
    assert stored[0]["notes"] == "upgrade driven by thermal overloads."


def test_write_extract_records_null_cost_when_the_model_found_nothing(panel) -> None:
    write_extract(StudyExtract(queue_id="ac2115"))
    rows = query_projects("SELECT cost_per_kw FROM projects WHERE queue_id = 'ac2115'")
    assert rows[0]["cost_per_kw"] is None


def test_already_extracted_queue_ids_is_empty_before_the_first_write(panel) -> None:
    assert already_extracted_queue_ids() == set()


def test_already_extracted_queue_ids_reflects_a_stored_extract(panel) -> None:
    write_extract(StudyExtract(queue_id="ac2115", studied_mw=100.0))
    assert already_extracted_queue_ids() == {"ac2115"}


def test_dry_run_extract_all_reports_counts_and_never_constructs_a_real_client(
    store_with_chunks, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "dry_run", True)
    monkeypatch.setattr(settings, "openai_api_key", "fake-key-for-test")

    def _blow_up(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry run must not construct a real client")

    monkeypatch.setattr("src.extract.ChatOpenAI", _blow_up)
    monkeypatch.setattr("src.extract.OpenAIEmbeddings", _blow_up)

    summary = extract_all()
    assert summary["projects"] == 1
    assert summary["input_tokens"] > 0


def test_rebuild_reprocesses_a_project_that_already_has_a_stored_extract(
    store_with_chunks, monkeypatch
) -> None:
    write_extract(StudyExtract(queue_id="ac2115", studied_mw=100.0))
    monkeypatch.setattr(settings, "dry_run", True)
    monkeypatch.setattr(settings, "openai_api_key", "fake-key-for-test")

    skipped = extract_all()
    assert skipped["projects"] == 0

    reprocessed = extract_all(rebuild=True)
    assert reprocessed["projects"] == 1


def test_find_lbnl_workbook_returns_none_when_the_directory_is_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.queue_raw_dir.mkdir(parents=True, exist_ok=True)
    assert find_lbnl_workbook() is None


def test_load_lbnl_reference_is_none_when_the_cost_column_is_absent(tmp_path) -> None:
    path = tmp_path / "fake_lbnl.xlsx"
    frame = pd.DataFrame({"q_id": ["AC2-115"], "project_name": ["Gray Solar"]})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=LBNL_SHEET, index=False, startrow=1)
    assert load_lbnl_reference(path) is None


def test_run_validation_prints_a_clear_message_when_no_workbook_is_present(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.queue_raw_dir.mkdir(parents=True, exist_ok=True)
    run_validation()
    assert "skipping validation" in capsys.readouterr().out
