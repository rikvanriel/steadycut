"""Where per-camera measured data lives -- outside the source tree.

Measured constants belong to the user's camera body, not to the code. An rpm
or deb install ships a read-only source tree, and a value discovered on first
use has to survive package upgrades, so measurements live in the user's config
directory:

    $XDG_CONFIG_HOME/steadycut/cameras.json   (default ~/.config/steadycut)

The file is plain JSON, keyed by trailer serial number (or by model name for
files that carry no trailer), so it can be inspected, edited, backed up, or
shipped as a distribution's default profile without touching the code.

Derived, recomputable data -- the per-file identity parse cache -- lives in the
cache directory instead ($XDG_CACHE_HOME/steadycut/files.json), because a cache
must never be mistaken for a measurement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP = "steadycut"


def _xdg(env: str, default: str) -> Path:
    base = os.environ.get(env)
    return Path(base) if base else Path.home() / default


def config_dir() -> Path:
    """$XDG_CONFIG_HOME/steadycut (default ~/.config/steadycut)."""
    return _xdg("XDG_CONFIG_HOME", ".config") / APP


def cache_dir() -> Path:
    """$XDG_CACHE_HOME/steadycut (default ~/.cache/steadycut)."""
    return _xdg("XDG_CACHE_HOME", ".cache") / APP


def registry_path() -> Path:
    """The per-camera measured-data file."""
    return config_dir() / "cameras.json"


def file_cache_path() -> Path:
    """The per-file identity parse cache."""
    return cache_dir() / "files.json"


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    staged.replace(path)


def load_registry() -> dict:
    """Every measured camera in the user's config, keyed by serial."""
    return _load(registry_path())


def entry(key: str) -> dict | None:
    """One camera's measured data, or None when it has never been measured."""
    return load_registry().get(key)


def upsert(key: str, fields: dict) -> dict:
    """Merge `fields` into a camera's entry and write the registry back."""
    data = load_registry()
    item = dict(data.get(key) or {})
    item.update(fields)
    data[key] = item
    _write(registry_path(), data)
    return item


def load_file_cache() -> dict:
    return _load(file_cache_path())


def store_file_cache(key: str, meta: dict) -> None:
    data = load_file_cache()
    data[key] = meta
    _write(file_cache_path(), data)
