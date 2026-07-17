"""Build the structured queue panel in DuckDB.

Reads interconnection queue data from one of three sources and writes a single
projects table to the configured DuckDB path, one row per project. The default
source is the LBNL Queued Up workbook because it needs no API key and pins a
reproducible snapshot. The gridstatus source pulls the live PJM queue instead,
and the fixture source reads the committed sample so the tests run with no
network.

The panel is the only thing the agent's structured tool needs from this stage.
That tool imports query_projects from here, which is why the query path is read
only and refuses anything that is not a single select.
"""

import argparse
import re
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from src.config import settings

# the LBNL data sits on this sheet. the sheet opens with a banner row, so the real
# column header is the second row.
LBNL_SHEET = "03. Complete Queue Data"
LBNL_HEADER_ROW = 1

# the canonical raw layout every reader returns, so one normalizer serves all sources.
SOURCE_COLUMNS = [
    "queue_id",
    "project_name",
    "capacity_mw",
    "fuel_type",
    "status",
    "request_date",
    "in_service_date",
    "county",
    "state",
    "poi",
]

LBNL_COLUMNS = {
    "q_id": "queue_id",
    "project_name": "project_name",
    "mw_1": "capacity_mw",
    "type_clean": "fuel_type",
    "q_status": "status",
    "q_date": "request_date",
    "prop_date": "in_service_date",
    "county": "county",
    "state": "state",
    "poi_name": "poi",
}

GRIDSTATUS_COLUMNS = {
    "Queue ID": "queue_id",
    "Project Name": "project_name",
    "Capacity (MW)": "capacity_mw",
    "Generation Type": "fuel_type",
    "Status": "status",
    "Queue Date": "request_date",
    "Proposed Completion Date": "in_service_date",
    "County": "county",
    "State": "state",
    "Interconnection Location": "poi",
}

# raw status spellings mapped to the five allowed values. anything absent becomes other,
# so unusual states are kept rather than dropped.
STATUS_MAP = {
    "active": "active",
    "withdrawn": "withdrawn",
    "operational": "operational",
    "suspended": "suspended",
    "in service": "operational",
}

# raw fuel spellings mapped to the controlled list. anything absent becomes other.
FUEL_MAP = {
    "solar": "solar",
    "photovoltaic": "solar",
    "pv": "solar",
    "wind": "wind",
    "offshore wind": "wind",
    "battery": "storage",
    "bess": "storage",
    "solar+battery": "solar+storage",
    "gas": "gas",
    "natural gas": "gas",
    "hydro": "hydro",
}

CREATE_PROJECTS_TABLE = """
    CREATE TABLE projects (
        queue_id TEXT PRIMARY KEY,
        project_name TEXT,
        iso TEXT NOT NULL,
        capacity_mw DOUBLE,
        fuel_type TEXT NOT NULL,
        status TEXT NOT NULL,
        request_date DATE,
        in_service_date DATE,
        county TEXT,
        state TEXT,
        poi TEXT,
        cost_per_kw DOUBLE,
        queue_age_days INTEGER,
        is_withdrawn BOOLEAN NOT NULL
    )
"""

# cost_per_kw is filled later by the extraction stage, so it loads as null here.
# queue_age_days and is_withdrawn are derived from the normalized fields at load.
INSERT_PROJECTS = """
    INSERT INTO projects
    SELECT
        queue_id,
        project_name,
        ? AS iso,
        capacity_mw,
        fuel_type,
        status,
        request_date,
        in_service_date,
        county,
        state,
        poi,
        CAST(NULL AS DOUBLE) AS cost_per_kw,
        CASE
            WHEN request_date IS NULL THEN NULL
            ELSE date_diff('day', request_date, current_date)
        END AS queue_age_days,
        status = 'withdrawn' AS is_withdrawn
    FROM incoming
"""


def clean_text(value: object) -> str | None:
    """Return a trimmed string, or None for blanks and missing values."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def parse_capacity(value: object) -> float | None:
    """Read capacity as MW, stripping commas and units. Returns None when it will not parse."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def parse_date(value: object) -> date | None:
    """Parse a date, returning None rather than raising when the value will not parse."""
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def normalize_status(value: object) -> str:
    """Map a raw status onto one of the five allowed values, defaulting to other."""
    text = clean_text(value)
    if text is None:
        return "other"
    return STATUS_MAP.get(text.lower(), "other")


def normalize_fuel(value: object) -> str:
    """Map a raw fuel or resource type onto the controlled list, defaulting to other."""
    text = clean_text(value)
    if text is None:
        return "other"
    return FUEL_MAP.get(text.lower(), "other")


def normalize_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the field level normalization to a raw source frame in the canonical layout."""
    frame = pd.DataFrame()
    frame["queue_id"] = raw["queue_id"].map(clean_text)
    frame["project_name"] = raw["project_name"].map(clean_text)
    frame["capacity_mw"] = raw["capacity_mw"].map(parse_capacity).astype("float64")
    frame["fuel_type"] = raw["fuel_type"].map(normalize_fuel)
    frame["status"] = raw["status"].map(normalize_status)
    frame["request_date"] = raw["request_date"].map(parse_date)
    frame["in_service_date"] = raw["in_service_date"].map(parse_date)
    frame["county"] = raw["county"].map(clean_text)
    frame["state"] = raw["state"].map(clean_text)
    frame["poi"] = raw["poi"].map(clean_text)
    return frame


def read_fixture() -> pd.DataFrame:
    """Read the committed sample as raw strings so the normalizer does all the work."""
    return pd.read_csv(settings.sample_queue_csv, dtype=str, keep_default_na=False)[
        SOURCE_COLUMNS
    ]


def read_lbnl(path: Path) -> pd.DataFrame:
    """Read the LBNL Queued Up workbook and return the in scope ISO rows in canonical layout."""
    raw = pd.read_excel(
        path,
        sheet_name=LBNL_SHEET,
        header=LBNL_HEADER_ROW,
        engine="openpyxl",
    )
    in_scope = raw[raw["region"] == settings.iso]
    return in_scope.rename(columns=LBNL_COLUMNS)[SOURCE_COLUMNS]


def read_gridstatus() -> pd.DataFrame:
    """Pull the live PJM queue through gridstatus in the canonical layout."""
    # imported here, only this source needs it and it pulls a heavy dependency tree.
    import gridstatus

    queue = gridstatus.PJM().get_interconnection_queue()
    return queue.rename(columns=GRIDSTATUS_COLUMNS)[SOURCE_COLUMNS]


def load_source(source: str, file: Path | None) -> pd.DataFrame:
    """Read the requested source into the canonical raw layout."""
    if source == "fixture":
        return read_fixture()
    if source == "gridstatus":
        return read_gridstatus()
    if source == "lbnl":
        if file is None:
            raise ValueError(
                "lbnl source needs --file pointing to the workbook in data/raw/queue"
            )
        return read_lbnl(file)
    raise ValueError(f"unknown source {source!r}")


def build_projects_table(frame: pd.DataFrame, db_path: Path) -> None:
    """Write the normalized frame to a fresh projects table, replacing any earlier build."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    try:
        connection.register("incoming", frame)
        connection.execute("DROP TABLE IF EXISTS projects")
        connection.execute(CREATE_PROJECTS_TABLE)
        connection.execute(INSERT_PROJECTS, [settings.iso])
    finally:
        connection.close()


def build_panel(source: str, file: Path | None) -> None:
    """Read the source, normalize it, and write the panel to the configured path."""
    raw = load_source(source, file)
    build_projects_table(normalize_frame(raw), settings.queue_db_path)


def _reject_if_not_select(sql: str) -> None:
    """Read only guard, the agent passes SQL so we reject anything that is not a single select."""
    statement = sql.strip().rstrip(";").strip()
    if ";" in statement:
        raise ValueError("only a single statement is allowed, remove the extra semicolon")
    if not statement.lower().startswith("select"):
        raise ValueError("only select statements are allowed")


def query_projects(sql: str) -> list[dict]:
    """Run a read only SQL query against the projects table and return rows as dicts."""
    _reject_if_not_select(sql)
    if not settings.queue_db_path.exists():
        raise FileNotFoundError(
            f"no panel at {settings.queue_db_path}, build it first with "
            "python -m src.ingest_queue"
        )
    connection = duckdb.connect(str(settings.queue_db_path), read_only=True)
    try:
        cursor = connection.execute(sql)
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        connection.close()


def format_summary() -> str:
    """Return a plain summary of the built panel, total rows and counts by status and fuel."""
    total = query_projects("SELECT count(*) AS n FROM projects")[0]["n"]
    status_rows = query_projects(
        "SELECT status, count(*) AS n FROM projects GROUP BY status ORDER BY n DESC, status"
    )
    fuel_rows = query_projects(
        "SELECT fuel_type, count(*) AS n FROM projects GROUP BY fuel_type ORDER BY n DESC, fuel_type"
    )
    lines = [f"projects: {total}", "", "by status:"]
    lines += [f"  {row['status']}: {row['n']}" for row in status_rows]
    lines += ["", "by fuel type:"]
    lines += [f"  {row['fuel_type']}: {row['n']}" for row in fuel_rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the interconnection queue panel in DuckDB."
    )
    parser.add_argument(
        "--source",
        choices=["lbnl", "gridstatus", "fixture"],
        default="lbnl",
        help="where to read the queue from, defaults to the LBNL workbook",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="path to the LBNL workbook, required for --source lbnl",
    )
    args = parser.parse_args()
    build_panel(args.source, args.file)
    print(format_summary())


if __name__ == "__main__":
    main()
