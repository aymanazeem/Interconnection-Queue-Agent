import shutil

import pytest

from src.config import settings
from src.ingest_queue import (
    build_projects_table,
    normalize_frame,
    normalize_fuel,
    query_projects,
    read_fixture,
)

# the schema the agent tool relies on, in declaration order.
EXPECTED_SCHEMA = [
    ("queue_id", "VARCHAR"),
    ("project_name", "VARCHAR"),
    ("iso", "VARCHAR"),
    ("capacity_mw", "DOUBLE"),
    ("fuel_type", "VARCHAR"),
    ("status", "VARCHAR"),
    ("request_date", "DATE"),
    ("in_service_date", "DATE"),
    ("county", "VARCHAR"),
    ("state", "VARCHAR"),
    ("poi", "VARCHAR"),
    ("cost_per_kw", "DOUBLE"),
    ("queue_age_days", "INTEGER"),
    ("is_withdrawn", "BOOLEAN"),
]


@pytest.fixture
def panel(tmp_path, monkeypatch):
    # redirect the data root to a sandbox so the test never touches the real panel, then
    # copy the committed fixture in so read_fixture still resolves through config.
    real_csv = settings.sample_queue_csv
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    settings.sample_queue_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(real_csv, settings.sample_queue_csv)
    build_projects_table(normalize_frame(read_fixture()), settings.queue_db_path)
    return settings.queue_db_path


def test_fixture_loads_expected_row_count(panel) -> None:
    rows = query_projects("SELECT count(*) AS n FROM projects")
    assert rows[0]["n"] == 15


def test_projects_schema_matches_contract(panel) -> None:
    rows = query_projects(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'projects' ORDER BY ordinal_position"
    )
    schema = [(row["column_name"], row["data_type"]) for row in rows]
    assert schema == EXPECTED_SCHEMA


def test_status_normalization_maps_in_service_to_operational(panel) -> None:
    rows = query_projects("SELECT status FROM projects WHERE queue_id = 'z014'")
    assert rows[0]["status"] == "operational"


def test_status_normalization_falls_back_to_other(panel) -> None:
    rows = query_projects("SELECT status FROM projects WHERE queue_id = 'z009'")
    assert rows[0]["status"] == "other"


def test_fuel_normalization_maps_photovoltaic_to_solar() -> None:
    assert normalize_fuel("Photovoltaic") == "solar"


def test_fuel_normalization_maps_battery_to_storage() -> None:
    assert normalize_fuel("battery") == "storage"


def test_query_projects_returns_dicts_and_keeps_the_real_pjm_id(panel) -> None:
    rows = query_projects("SELECT queue_id FROM projects WHERE queue_id = 'ac2115'")
    assert rows == [{"queue_id": "ac2115"}]


def test_query_projects_rejects_a_non_select() -> None:
    with pytest.raises(ValueError):
        query_projects("DROP TABLE projects")


def test_query_projects_rejects_stacked_statements() -> None:
    with pytest.raises(ValueError):
        query_projects("SELECT 1; DROP TABLE projects")


def test_unparseable_date_becomes_null(panel) -> None:
    rows = query_projects(
        "SELECT request_date, queue_age_days FROM projects WHERE queue_id = 'z015'"
    )
    assert rows[0]["request_date"] is None
    assert rows[0]["queue_age_days"] is None


def test_missing_capacity_becomes_null(panel) -> None:
    rows = query_projects("SELECT capacity_mw FROM projects WHERE queue_id = 'z013'")
    assert rows[0]["capacity_mw"] is None


def test_capacity_strips_commas(panel) -> None:
    rows = query_projects("SELECT capacity_mw FROM projects WHERE queue_id = 'z014'")
    assert rows[0]["capacity_mw"] == 1250.0


def test_is_withdrawn_tracks_status(panel) -> None:
    rows = query_projects(
        "SELECT is_withdrawn FROM projects WHERE queue_id IN ('z002', 'ac2115') ORDER BY queue_id"
    )
    assert [row["is_withdrawn"] for row in rows] == [False, True]
