"""Centralized project configuration and path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import tomllib
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


DEFAULT_DATA_FILE = "data/tracking_data.csv"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_DOCS_PPTX = "docs/browns_docs/performance_science_fellow_work_project.pptx"
DEFAULT_DATA_FALLBACKS = (
    "docs/browns_docs/tracking_data.csv",
    "browns_docs/tracking_data.csv",
)
DEFAULT_CONFIG_FILE = "config/browns_tracking.yaml"


class PathSettings(BaseModel):
    """File and directory locations for this project."""

    data_file: str = DEFAULT_DATA_FILE
    data_fallbacks: tuple[str, ...] = DEFAULT_DATA_FALLBACKS
    docs_pptx: str = DEFAULT_DOCS_PPTX
    output_dir: str = DEFAULT_OUTPUT_DIR

    @field_validator("data_file", "docs_pptx", "output_dir")
    @classmethod
    def _non_empty_path(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("path values must not be empty")
        return cleaned

    @field_validator("data_fallbacks", mode="before")
    @classmethod
    def _coerce_data_fallbacks(cls, value: Any) -> Any:
        if value is None:
            return ()
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("data_fallbacks")
    @classmethod
    def _strip_data_fallbacks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.strip() for item in value if item.strip())


class RuntimeSettings(BaseModel):
    """Runtime behavior controls for scripts and pipelines."""

    create_output_dirs: bool = False


class BrownsTrackingConfig(BaseModel):
    """Typed configuration model for project behavior."""

    model_config = ConfigDict(extra="ignore")
    paths: PathSettings = Field(default_factory=PathSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved canonical project paths."""

    project_root: Path
    data_file: Path
    output_dir: Path
    docs_pptx: Path
    data_fallbacks: tuple[Path, ...]


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by locating `pyproject.toml`."""
    env_root = os.getenv("BROWNS_TRACKING_PROJECT_ROOT")
    if env_root:
        return _resolve_path(Path(env_root), Path.cwd())

    cursor = (start or Path.cwd()).resolve()
    for candidate in (cursor, *cursor.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate

    module_cursor = Path(__file__).resolve()
    for candidate in (module_cursor, *module_cursor.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate

    raise FileNotFoundError("Could not find project root containing pyproject.toml")


@lru_cache(maxsize=1)
def default_project_config() -> BrownsTrackingConfig:
    """Load config with OmegaConf merge + Pydantic validation."""
    project_root = find_project_root()
    merged = _load_merged_config(project_root)
    try:
        return BrownsTrackingConfig.model_validate(merged)
    except ValidationError as exc:
        raise ValueError(f"Invalid browns_tracking config: {exc}") from exc


@lru_cache(maxsize=1)
def default_project_paths() -> ProjectPaths:
    """Resolve canonical paths from validated project config."""
    project_root = find_project_root()
    config = default_project_config()
    path_cfg = config.paths

    data_candidates: list[Path] = [_resolve_path(Path(path_cfg.data_file), project_root)]
    fallback_paths = tuple(
        _resolve_path(Path(fallback), project_root) for fallback in path_cfg.data_fallbacks
    )
    data_candidates.extend(fallback_paths)

    selected_data_file = _first_existing(data_candidates) or data_candidates[0]
    output_dir = _resolve_path(Path(path_cfg.output_dir), project_root)
    docs_pptx = _resolve_path(Path(path_cfg.docs_pptx), project_root)

    if config.runtime.create_output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        project_root=project_root,
        data_file=selected_data_file,
        output_dir=output_dir,
        docs_pptx=docs_pptx,
        data_fallbacks=fallback_paths,
    )


def resolve_data_file(input_path: str | Path | None = None) -> Path:
    """Resolve an explicit or default tracking-data file path."""
    paths = default_project_paths()
    if input_path is None:
        return paths.data_file
    return _resolve_path(Path(input_path), paths.project_root)


def resolve_output_dir(output_dir: str | Path | None = None) -> Path:
    """Resolve an explicit or default output directory path."""
    paths = default_project_paths()
    if output_dir is None:
        return paths.output_dir
    return _resolve_path(Path(output_dir), paths.project_root)


def clear_project_path_cache() -> None:
    """Clear cached config; useful for tests or env-var changes."""
    default_project_config.cache_clear()
    default_project_paths.cache_clear()


def _load_merged_config(project_root: Path) -> dict[str, Any]:
    base_cfg = {
        "paths": {
            "data_file": DEFAULT_DATA_FILE,
            "data_fallbacks": list(DEFAULT_DATA_FALLBACKS),
            "docs_pptx": DEFAULT_DOCS_PPTX,
            "output_dir": DEFAULT_OUTPUT_DIR,
        },
        "runtime": {
            "create_output_dirs": False,
        },
    }
    pyproject_paths = _load_pyproject_paths(project_root)
    pyproject_cfg: dict[str, Any] = {"paths": pyproject_paths} if pyproject_paths else {}

    merged = OmegaConf.merge(
        base_cfg,
        pyproject_cfg,
        _load_file_config(project_root),
        _load_env_overrides(),
    )
    raw = OmegaConf.to_container(merged, resolve=True)
    return raw if isinstance(raw, dict) else {}


def _load_file_config(project_root: Path) -> dict[str, Any]:
    env_path = os.getenv("BROWNS_TRACKING_CONFIG_FILE")
    if env_path:
        cfg_path = _resolve_path(Path(env_path), project_root)
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"BROWNS_TRACKING_CONFIG_FILE points to missing file: {cfg_path}"
            )
    else:
        cfg_path = project_root / DEFAULT_CONFIG_FILE
        if not cfg_path.exists():
            return {}

    loaded = OmegaConf.load(cfg_path)
    raw = OmegaConf.to_container(loaded, resolve=True)
    return raw if isinstance(raw, dict) else {}


def _load_env_overrides() -> dict[str, Any]:
    paths: dict[str, Any] = {}
    if env_data := os.getenv("BROWNS_TRACKING_DATA_FILE"):
        paths["data_file"] = env_data
    if env_fallbacks := os.getenv("BROWNS_TRACKING_DATA_FALLBACKS"):
        paths["data_fallbacks"] = _split_csv(env_fallbacks)
    if env_output := os.getenv("BROWNS_TRACKING_OUTPUT_DIR"):
        paths["output_dir"] = env_output
    if env_docs := os.getenv("BROWNS_TRACKING_DOCS_PPTX"):
        paths["docs_pptx"] = env_docs

    runtime: dict[str, Any] = {}
    if env_create_output := os.getenv("BROWNS_TRACKING_CREATE_OUTPUT_DIRS"):
        runtime["create_output_dirs"] = _parse_env_bool(env_create_output)

    overrides: dict[str, Any] = {}
    if paths:
        overrides["paths"] = paths
    if runtime:
        overrides["runtime"] = runtime
    return overrides


def _resolve_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (project_root / path).resolve()


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_env_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "BROWNS_TRACKING_CREATE_OUTPUT_DIRS must be one of: "
        "1,true,yes,on,0,false,no,off"
    )


def _load_pyproject_paths(project_root: Path) -> dict[str, Any]:
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        return {}

    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)

    tool_cfg = pyproject.get("tool", {})
    browns_cfg = tool_cfg.get("browns_tracking", {})
    paths_cfg = browns_cfg.get("paths", {})
    return paths_cfg if isinstance(paths_cfg, dict) else {}
