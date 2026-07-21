"""Command line entry point for the queue intelligence agent.

Three subcommands. setup-check inspects the local environment and makes no api call, so it
is safe to run first on a fresh clone. ask answers one question. chat opens a session that
carries context across questions. The reasoning lives in the agent and ingest modules, this
file parses arguments, routes to them, and prints.
"""

import argparse

from src.agent import Conversation, ask
from src.config import settings
from src.ingest_pdfs import existing_queue_ids, get_querying_store, open_store
from src.ingest_queue import query_projects

# chroma writes this file at the root of its persist directory, so its presence marks a built store.
CHROMA_DB_FILE = "chroma.sqlite3"

EXIT_WORDS = {"exit", "quit"}


def _key_status() -> str:
    """Whether an api key is loaded, without printing the key itself."""
    return "present" if settings.openai_api_key else "missing, set OPENAI_API_KEY in .env"


def _panel_status() -> str:
    """The panel readiness line, plus how many projects it holds once it exists."""
    if not settings.queue_db_path.exists():
        return "missing, build it with python -m src.ingest_queue"
    count = query_projects("SELECT count(*) AS n FROM projects")[0]["n"]
    return f"present, {count} projects"


def _vector_store_status() -> str:
    """The vector store readiness line, plus how many studies are embedded in it."""
    if not (settings.chroma_dir / CHROMA_DB_FILE).exists():
        return "missing, build it with python -m src.ingest_pdfs"
    studies = existing_queue_ids(open_store(embeddings=None))
    return f"present, {len(studies)} studies embedded"


def run_setup_check() -> None:
    """Print key, model, and data readiness so a new user can see what is and is not ready."""
    print("setup check")
    print(f"  openai key: {_key_status()}")
    print(f"  dry run: {settings.dry_run}")
    print(f"  chat model: {settings.chat_model}")
    print(f"  extract model: {settings.extract_model}")
    print(f"  embed model: {settings.embed_model}")
    print(f"  queue panel: {_panel_status()}")
    print(f"  vector store: {_vector_store_status()}")


def run_ask(question: str) -> None:
    """Answer one question and print it."""
    print(ask(question))


def run_chat() -> None:
    """Open a read eval print loop that carries context until the user exits."""
    if settings.dry_run:
        print("dry run, answers are canned and no api call is made. set DRY_RUN=false for real answers.")
    else:
        # build the chroma client on the main thread before the agent uses it from worker threads.
        get_querying_store()
    print("ask a question, or type exit to quit.")
    conversation = Conversation()
    while True:
        try:
            question = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            # ctrl d or ctrl c leaves the loop cleanly rather than dumping a traceback.
            print()
            break
        if question.lower() in EXIT_WORDS:
            break
        if not question:
            continue
        print(conversation.ask(question))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the interconnection queue intelligence agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("setup-check", help="check the key, models, and data before asking anything")
    ask_parser = subparsers.add_parser("ask", help="answer one question and exit")
    ask_parser.add_argument("question", help="the question to ask, in plain english")
    subparsers.add_parser("chat", help="ask several questions in one session, carrying context")

    args = parser.parse_args()
    if args.command == "setup-check":
        run_setup_check()
    elif args.command == "ask":
        run_ask(args.question)
    elif args.command == "chat":
        run_chat()


if __name__ == "__main__":
    main()
