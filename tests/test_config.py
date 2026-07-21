import pytest

from src.config import Settings, settings, token_cost


def test_settings_instance_is_available() -> None:
    assert isinstance(settings, Settings)


def test_model_names_are_non_empty() -> None:
    assert settings.chat_model
    assert settings.extract_model
    assert settings.embed_model


def test_all_paths_sit_under_the_data_root() -> None:
    data_root = settings.data_dir
    paths = [
        settings.queue_raw_dir,
        settings.pdf_raw_dir,
        settings.processed_dir,
        settings.queue_db_path,
        settings.vectors_dir,
        settings.chroma_dir,
        settings.fixtures_dir,
        settings.sample_queue_csv,
        settings.sample_study_pdf,
    ]
    for path in paths:
        assert data_root in path.parents


def test_dry_run_defaults_to_true(monkeypatch: pytest.MonkeyPatch) -> None:
    # a fresh clone must not spend money. clearing the env and skipping .env asserts the
    # built in default rather than whatever the developer has set locally.
    monkeypatch.delenv("DRY_RUN", raising=False)
    assert Settings(_env_file=None).dry_run is True


def test_max_pdfs_defaults_to_twenty_five(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_PDFS", raising=False)
    assert Settings(_env_file=None).max_pdfs == 25


def test_output_dirs_exist_after_load() -> None:
    # config creates these on load so other modules do not have to.
    assert settings.processed_dir.is_dir()
    assert settings.vectors_dir.is_dir()


def test_search_k_reads_at_least_as_wide_as_extraction_retrieval() -> None:
    # the narrative search tool answers open ended questions, so it should not retrieve narrower
    # than the focused extraction pass.
    assert settings.search_k >= settings.retrieval_k


def test_token_cost_reads_the_price_from_the_model_table() -> None:
    assert token_cost("gpt-4.1-mini", 1_000_000, 0) == pytest.approx(0.40)
    assert token_cost("gpt-4.1-mini", 0, 1_000_000) == pytest.approx(1.60)


def test_token_cost_raises_for_a_model_with_no_listed_price() -> None:
    with pytest.raises(KeyError):
        token_cost("some-unlisted-model", 100, 100)


def test_every_configured_model_has_a_price() -> None:
    # each model the pipeline actually runs must be priced, so cost reporting never guesses.
    for model in (settings.chat_model, settings.extract_model, settings.embed_model):
        assert token_cost(model, 1, 1) >= 0
