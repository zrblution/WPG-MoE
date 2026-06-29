"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT.parent


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml_file(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_base_config_path(config_path: Path, base_config: str | Path) -> Path:
    base_path = Path(base_config)
    if base_path.is_absolute():
        return base_path
    return (config_path.parent / base_path).resolve()


def _load_explicit_config_chain(config_path: Path, visited: Tuple[Path, ...]) -> tuple[Dict[str, Any], Tuple[Path, ...]]:
    resolved_path = config_path.resolve()
    if resolved_path in visited:
        cycle = " -> ".join(str(path) for path in [*visited, resolved_path])
        raise ValueError(f"Detected circular base_config reference: {cycle}")
    visited = (*visited, resolved_path)

    config = _read_yaml_file(resolved_path)
    base_ref = config.pop("base_config", None)
    if base_ref is None:
        return config, visited

    base_path = _resolve_base_config_path(resolved_path, base_ref)
    if not base_path.exists():
        raise FileNotFoundError(f"base_config not found: {base_path}")
    base_config, visited = _load_explicit_config_chain(base_path, visited)
    return merge_dicts(base_config, config), visited


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config and merge it with the default config if needed."""
    config_path = Path(config_path).resolve()
    default_path = CODE_ROOT / "configs" / "default.yaml"
    config, visited = _load_explicit_config_chain(config_path, visited=())
    if config_path == default_path.resolve() or not default_path.exists():
        return config
    if default_path.resolve() in visited:
        return config
    default_cfg = _read_yaml_file(default_path)
    return merge_dicts(default_cfg, config)


def resolve_path(path_value: str | Path | None) -> Path | None:
    """Resolve a path relative to the code root."""
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return CODE_ROOT / path
