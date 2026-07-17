import pytest

from src.config import Settings, settings


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
