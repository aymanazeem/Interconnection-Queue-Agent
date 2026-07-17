import shutil
from unittest.mock import MagicMock

import httpx
import pytest

from src.config import REPO_ROOT, settings
from src.fetch_pdfs import build_study_urls, fetch_study_pdf, normalize_pdf_id, verify_local_pdfs


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
