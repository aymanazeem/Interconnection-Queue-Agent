import shutil
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest

from src.config import REPO_ROOT, settings
from src.fetch_pdfs import (
    build_study_urls,
    fetch_study_pdf,
    normalize_pdf_id,
    select_candidate_queue_ids,
    verify_local_pdfs,
)
from src.ingest_queue import SOURCE_COLUMNS, build_projects_table, normalize_frame


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    # redirect the data root to a sandbox so the test never touches the real pdf directory.
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.pdf_raw_dir.mkdir(parents=True, exist_ok=True)
    return settings.pdf_raw_dir


def test_normalize_pdf_id_lowercases_and_strips_the_hyphen() -> None:
    assert normalize_pdf_id("AC2-115") == "ac2115"


def test_build_study_urls_tries_the_impact_study_first() -> None:
    impact_url, feasibility_url = build_study_urls("AC2-115")
    assert impact_url == (
        "https://www.pjm.com/pjmfiles/pub/planning/project-queues/"
        "impact_studies/ac2115_imp.pdf"
    )
    assert feasibility_url == (
        "https://www.pjm.com/pjmfiles/pub/planning/project-queues/feas_docs/ac2115_fea.pdf"
    )


def test_offline_verification_reads_the_committed_fixture_pdf(sandbox) -> None:
    # data_dir is already redirected to the sandbox, so read the real fixture off repo root.
    real_pdf = REPO_ROOT / "data" / "fixtures" / "sample_study.pdf"
    if not real_pdf.exists():
        pytest.skip("sample_study.pdf fixture is not committed")
    shutil.copy(real_pdf, sandbox / "ac2115.pdf")
    summary = verify_local_pdfs()
    assert summary == {"checked": 1, "readable": 1}


def test_offline_verification_flags_an_unreadable_file(sandbox) -> None:
    (sandbox / "z001.pdf").write_bytes(b"not a real pdf")
    summary = verify_local_pdfs()
    assert summary == {"checked": 1, "readable": 0}


def test_fetch_study_pdf_skips_the_network_when_the_file_already_exists(sandbox) -> None:
    (sandbox / "ac2115.pdf").write_bytes(b"%PDF-1.4 fixture stand in")
    client = MagicMock(spec=httpx.Client)
    client.get.side_effect = AssertionError("should not fetch when the file already exists")
    assert fetch_study_pdf("ac2115", client) is True
    client.get.assert_not_called()


def _study_era_row(queue_id: str, request_date: str, status: str) -> dict:
    # a minimal panel row in the source layout, only the fields the selection reads matter.
    return {
        "queue_id": queue_id,
        "project_name": queue_id,
        "capacity_mw": "100",
        "fuel_type": "solar",
        "status": status,
        "request_date": request_date,
        "in_service_date": "",
        "county": "",
        "state": "PA",
        "poi": "",
    }


def _build_panel(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
    build_projects_table(normalize_frame(frame), settings.queue_db_path)


def test_selection_excludes_projects_after_the_study_cutoff(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _build_panel(
        [
            _study_era_row("AH1-001", "2024-01-01", "active"),
            _study_era_row("AG1-010", "2020-09-30", "active"),
            _study_era_row("AF2-020", "2019-12-01", "withdrawn"),
            _study_era_row("AC2-115", "2016-12-01", "withdrawn"),
        ]
    )
    ids = select_candidate_queue_ids(10)
    assert "AH1-001" not in ids
    assert ids == ["AG1-010", "AF2-020", "AC2-115"]


def test_selection_respects_the_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    _build_panel(
        [
            _study_era_row("AG1-001", "2020-09-30", "active"),
            _study_era_row("AF2-002", "2019-11-01", "active"),
            _study_era_row("AE2-003", "2018-11-01", "withdrawn"),
        ]
    )
    assert len(select_candidate_queue_ids(2)) == 2
