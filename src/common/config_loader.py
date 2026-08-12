"""
src.common.config_loader

Phase: Phase 0
Purpose: Load and validate YAML configs, resolving the `extends:` key so that
    Phase 4 training configs can inherit from config/training/_shared_defaults.yaml
    without copy-paste. Deliberately dependency-light: standard library + PyYAML
    only, so it is importable during the Phase 0 import-smoke-test.

Merge semantics for `extends`:
    - The parent (extended) file is loaded first, then the child is deep-merged on
      top of it. Child scalars/lists overwrite parent; child dicts merge key-by-key.
    - `extends` may be a single path or a list of paths (applied left-to-right).
    - Paths in `extends` are resolved relative to the child file's directory.
    - `extends` is stripped from the returned config.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_MAX_EXTENDS_DEPTH = 10


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or fails validation."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: `override` deep-merged on top of `base`."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level config in {path} must be a mapping, got {type(data).__name__}")
    return data


def load_config(path: str | Path, _depth: int = 0) -> dict[str, Any]:
    """Load a YAML config, resolving `extends` inheritance.

    Args:
        path: path to the YAML config file.
        _depth: internal recursion guard; do not set.

    Returns:
        The fully-merged config dict with `extends` removed.
    """
    if _depth > _MAX_EXTENDS_DEPTH:
        raise ConfigError(
            f"extends chain exceeded max depth {_MAX_EXTENDS_DEPTH} at {path} "
            "(possible circular reference)"
        )

    path = Path(path)
    raw = _load_raw(path)

    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    if isinstance(extends, str):
        parents = [extends]
    elif isinstance(extends, list):
        parents = extends
    else:
        raise ConfigError(f"`extends` in {path} must be a string or list, got {type(extends).__name__}")

    merged: dict[str, Any] = {}
    for parent_ref in parents:
        parent_path = (path.parent / parent_ref).resolve()
        parent_cfg = load_config(parent_path, _depth=_depth + 1)
        merged = _deep_merge(merged, parent_cfg)

    return _deep_merge(merged, raw)


def require_keys(config: dict[str, Any], keys: list[str], *, context: str = "") -> None:
    """Raise ConfigError if any of `keys` is absent from `config`.

    Supports dotted paths, e.g. "arena.radius_m".
    """
    ctx = f" ({context})" if context else ""
    for key in keys:
        node: Any = config
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"Missing required config key '{key}'{ctx}")
            node = node[part]
