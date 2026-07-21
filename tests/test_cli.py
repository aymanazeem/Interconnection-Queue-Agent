import src.cli as cli
from src.config import settings


def _feed_inputs(monkeypatch, lines: list[str]) -> None:
    """Feed the chat loop a fixed list of lines, then signal end of input once they run out."""
    supplied = iter(lines)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(supplied)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)


def test_setup_check_reports_missing_pieces(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    cli.run_setup_check()
    out = capsys.readouterr().out
    assert "openai key: missing" in out
    assert "queue panel: missing" in out
    assert "vector store: missing" in out


def test_setup_check_reports_present_pieces_without_an_api_call(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "a-key")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    panel = tmp_path / "processed" / "queue.duckdb"
    panel.parent.mkdir(parents=True)
    panel.write_text("db")
    chroma = tmp_path / "vectors" / "chroma"
    chroma.mkdir(parents=True)
    (chroma / cli.CHROMA_DB_FILE).write_text("db")

    captured = {}

    def fake_open_store(embeddings=None):
        captured["embeddings"] = embeddings
        return "store"

    monkeypatch.setattr("src.cli.query_projects", lambda sql: [{"n": 7666}])
    monkeypatch.setattr("src.cli.open_store", fake_open_store)
    monkeypatch.setattr("src.cli.existing_queue_ids", lambda store: {"AF1-236", "ac2115"})

    cli.run_setup_check()
    out = capsys.readouterr().out
    assert "openai key: present" in out
    assert "queue panel: present, 7666 projects" in out
    assert "vector store: present, 2 studies embedded" in out
    # the store is opened with no embedding function, so inspecting it makes no api call.
    assert captured["embeddings"] is None


def test_ask_prints_the_agent_answer(monkeypatch, capsys) -> None:
    monkeypatch.setattr("src.cli.ask", lambda question: f"answer to {question}")
    cli.run_ask("how many solar projects")
    assert "answer to how many solar projects" in capsys.readouterr().out


def test_ask_in_dry_run_prints_the_canned_reply(monkeypatch, capsys) -> None:
    monkeypatch.setattr(settings, "dry_run", True)
    cli.run_ask("which projects are at risk")
    out = capsys.readouterr().out
    assert "dry run" in out
    assert "which projects are at risk" in out


def test_chat_answers_each_turn_then_exits_on_the_exit_word(monkeypatch, capsys) -> None:
    monkeypatch.setattr(settings, "dry_run", True)
    _feed_inputs(monkeypatch, ["how many solar projects", "exit"])
    cli.run_chat()
    out = capsys.readouterr().out
    assert "how many solar projects" in out
    # one question answered, the exit word itself is not answered.
    assert out.count("no model call was made") == 1


def test_chat_skips_blank_lines(monkeypatch, capsys) -> None:
    monkeypatch.setattr(settings, "dry_run", True)
    _feed_inputs(monkeypatch, ["", "   ", "exit"])
    cli.run_chat()
    assert "no model call was made" not in capsys.readouterr().out


def test_chat_exits_on_end_of_input(monkeypatch, capsys) -> None:
    monkeypatch.setattr(settings, "dry_run", True)

    def raise_eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    cli.run_chat()
    assert "ask a question" in capsys.readouterr().out
