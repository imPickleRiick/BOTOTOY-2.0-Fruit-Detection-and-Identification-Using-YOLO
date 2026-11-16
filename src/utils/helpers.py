"""General helpers shared by all modules: config loading, paths, and JSON I/O.

All project paths are resolved relative to the project root, so the code works on
any machine (Windows laptop, Kaggle, Colab) without hardcoded absolute paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

# src/utils/helpers.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load the YAML configuration file (defaults to config/config.yaml)."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Turn a project-relative path (from the config) into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return p


def load_json(path: str | Path) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def list_images(folder: str | Path, extensions: Iterable[str]) -> list[Path]:
    """Return image files in a folder (non-recursive), sorted for reproducibility."""
    exts = {e.lower() for e in extensions}
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def relpath_posix(path: Path, start: Path) -> str:
    """Relative path with forward slashes (portable between Windows and Linux)."""
    return path.relative_to(start).as_posix()
