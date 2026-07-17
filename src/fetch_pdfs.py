"""Download PJM study report PDFs, keyed to the queue panel.

Reads a small, capped set of queue ids from the panel, builds the two
candidate PJM study urls for each, and saves whichever one exists to
data/raw/pdfs/{queue_id}.pdf. Most panel ids have no downloadable study. That is
normal, so a missing pdf is a quiet skip, not an error.

Panel ids are stored uppercase and hyphenated, for example AC2-115, but the PJM
urls use a lowercase, unpunctuated form, ac2115. normalize_pdf_id bridges the two,
and the saved file keeps the panel's own queue_id so later stages can map it back.

An --offline mode covers the case where the network is unavailable. It skips PJM
entirely and just confirms that whatever pdfs already sit in data/raw/pdfs are
readable, which is also what the automatic test exercises.
"""

import argparse
import re
import time
from pathlib import Path

import fitz
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from src.config import settings
from src.ingest_queue import query_projects

IMPACT_STUDY_URL = (
    "https://www.pjm.com/pjmfiles/pub/planning/project-queues/"
    "impact_studies/{pdf_id}_imp.pdf"
)
FEASIBILITY_STUDY_URL = (
    "https://www.pjm.com/pjmfiles/pub/planning/project-queues/"
    "feas_docs/{pdf_id}_fea.pdf"
)

# the pdf file signature, used to reject html error pages that answer with a 200.
PDF_MAGIC = b"%PDF"

# fetch tuning. retries cover transient connection drops, not missing files. the timeout is
# generous because pjm's file server is slow and some studies run to several MB.
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_FETCH_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 2

# individual study PDFs exist only for the pre cluster era. the selection takes the most recent
# projects before the cutoff, ties broken by queue_id so the same sample comes back each run.
CANDIDATE_IDS_SQL = (
    "SELECT queue_id FROM projects "
    "WHERE status IN ('active', 'withdrawn') "
    "AND request_date IS NOT NULL "
    "AND request_date < DATE '{cutoff}' "
    "ORDER BY request_date DESC, queue_id "
    "LIMIT {limit}"
)


def normalize_pdf_id(queue_id: str) -> str:
    """Return the lowercase unpunctuated id the pjm urls use, AC2-115 becomes ac2115."""
    return re.sub(r"[^a-z0-9]", "", queue_id.lower())


def build_study_urls(queue_id: str) -> tuple[str, str]:
    """Return the impact and feasibility study urls for a queue id, impact study first."""
    pdf_id = normalize_pdf_id(queue_id)
    return IMPACT_STUDY_URL.format(pdf_id=pdf_id), FEASIBILITY_STUDY_URL.format(pdf_id=pdf_id)


def select_candidate_queue_ids(limit: int) -> list[str]:
    """Return up to limit study era queue ids, most recent active and withdrawn projects first."""
    sql = CANDIDATE_IDS_SQL.format(cutoff=settings.pdf_study_cutoff.isoformat(), limit=int(limit))
    return [row["queue_id"] for row in query_projects(sql)]


@retry(
    stop=stop_after_attempt(MAX_FETCH_ATTEMPTS),
    wait=wait_fixed(RETRY_WAIT_SECONDS),
    retry=retry_if_exception_type(httpx.TransportError),
    reraise=True,
)
def _get_pdf_bytes(client: httpx.Client, url: str) -> bytes | None:
    """Fetch one url and return its bytes only when the response is actually a pdf."""
    response = client.get(url)
    if response.status_code != 200:
        return None
    if not response.content.startswith(PDF_MAGIC):
        return None
    return response.content


def fetch_study_pdf(queue_id: str, client: httpx.Client) -> bool:
    """Ensure the study pdf for one queue id sits on disk. Returns whether one is present."""
    destination = settings.pdf_raw_dir / f"{queue_id}.pdf"
    if destination.exists():
        return True
    for url in build_study_urls(queue_id):
        try:
            content = _get_pdf_bytes(client, url)
        except httpx.HTTPError:
            # the environment may block outbound requests entirely, treat that like a miss.
            content = None
        time.sleep(settings.pdf_fetch_delay_seconds)
        if content is not None:
            destination.write_bytes(content)
            return True
    return False


def fetch_all(max_pdfs: int) -> dict[str, int]:
    """Fetch up to max_pdfs studies for the most recent active and withdrawn projects."""
    settings.pdf_raw_dir.mkdir(parents=True, exist_ok=True)
    queue_ids = select_candidate_queue_ids(max_pdfs)
    fetched = 0
    headers = {"User-Agent": settings.pdf_user_agent}
    with httpx.Client(headers=headers, timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False) as client:
        for queue_id in queue_ids:
            found = fetch_study_pdf(queue_id, client)
            print(f"{queue_id}: {'fetched' if found else 'skipped'}")
            if found:
                fetched += 1
    return {"tried": len(queue_ids), "fetched": fetched, "skipped": len(queue_ids) - fetched}


def is_readable_pdf(path: Path) -> bool:
    """Open a pdf and return whether it parses and has at least one page."""
    try:
        document = fitz.open(path)
    except fitz.FileDataError:
        return False
    try:
        return document.page_count > 0
    finally:
        document.close()


def verify_local_pdfs() -> dict[str, int]:
    """Check every pdf already in the pdf directory and report how many are readable."""
    paths = sorted(settings.pdf_raw_dir.glob("*.pdf"))
    readable = 0
    for path in paths:
        ok = is_readable_pdf(path)
        print(f"{path.name}: {'readable' if ok else 'unreadable'}")
        if ok:
            readable += 1
    return {"checked": len(paths), "readable": readable}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PJM study report PDFs for the queue panel."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the network and verify the pdfs already on disk instead",
    )
    args = parser.parse_args()
    if args.offline:
        summary = verify_local_pdfs()
        print(f"checked {summary['checked']}, readable {summary['readable']}")
        return
    summary = fetch_all(settings.max_pdfs)
    print(f"tried {summary['tried']}, fetched {summary['fetched']}, skipped {summary['skipped']}")


if __name__ == "__main__":
    main()
