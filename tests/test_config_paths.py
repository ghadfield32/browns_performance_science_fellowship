from __future__ import annotations

from pathlib import Path

from browns_tracking.config import (
    default_project_config,
    clear_project_path_cache,
    default_project_paths,
    resolve_data_file,
    resolve_output_dir,
)


def test_default_data_path_prefers_data_folder() -> None:
    clear_project_path_cache()
    paths = default_project_paths()
    assert paths.data_file.as_posix().endswith("data/tracking_data.csv")
    assert paths.data_file.exists()


def test_env_override_for_data_file(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom_tracking_data.csv"
    custom.write_text("ts,x,y\n", encoding="utf-8")

    monkeypatch.setenv("BROWNS_TRACKING_DATA_FILE", str(custom))
    clear_project_path_cache()

    resolved = resolve_data_file()
    assert resolved == custom.resolve()

    monkeypatch.delenv("BROWNS_TRACKING_DATA_FILE", raising=False)
    clear_project_path_cache()


def test_env_override_for_output_dir(monkeypatch, tmp_path: Path) -> None:
    custom_output = tmp_path / "exports"
    monkeypatch.setenv("BROWNS_TRACKING_OUTPUT_DIR", str(custom_output))
    clear_project_path_cache()

    resolved = resolve_output_dir()
    assert resolved == custom_output.resolve()

    monkeypatch.delenv("BROWNS_TRACKING_OUTPUT_DIR", raising=False)
    clear_project_path_cache()


def test_external_yaml_config_file_override(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "yaml_tracking_data.csv"
    custom.write_text("ts,x,y\n", encoding="utf-8")
    custom_output = tmp_path / "yaml_exports"
    custom_docs = tmp_path / "brief.pptx"
    custom_docs.write_text("placeholder", encoding="utf-8")
    cfg = tmp_path / "browns_tracking.yaml"
    cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  data_file: {custom.as_posix()}",
                "  data_fallbacks: []",
                f"  output_dir: {custom_output.as_posix()}",
                f"  docs_pptx: {custom_docs.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("BROWNS_TRACKING_CONFIG_FILE", str(cfg))
    clear_project_path_cache()

    model = default_project_config()
    paths = default_project_paths()
    assert model.paths.data_file == custom.as_posix()
    assert paths.data_file == custom.resolve()
    assert paths.output_dir == custom_output.resolve()
    assert paths.docs_pptx == custom_docs.resolve()

    monkeypatch.delenv("BROWNS_TRACKING_CONFIG_FILE", raising=False)
    clear_project_path_cache()


def test_create_output_dirs_runtime_flag(monkeypatch, tmp_path: Path) -> None:
    custom_output = tmp_path / "auto_created_outputs"
    monkeypatch.setenv("BROWNS_TRACKING_OUTPUT_DIR", str(custom_output))
    monkeypatch.setenv("BROWNS_TRACKING_CREATE_OUTPUT_DIRS", "true")
    clear_project_path_cache()

    paths = default_project_paths()
    assert paths.output_dir == custom_output.resolve()
    assert custom_output.exists()

    monkeypatch.delenv("BROWNS_TRACKING_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("BROWNS_TRACKING_CREATE_OUTPUT_DIRS", raising=False)
    clear_project_path_cache()
