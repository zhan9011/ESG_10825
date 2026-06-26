from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into base."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file, including a single optional extends entry."""
    with path.open("r", encoding="utf-8") as source:
        data = yaml.safe_load(source) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent.parent / parent_path
        return deep_update(load_config(parent_path), data)
    return data


def set_config_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a config value using dot notation."""
    node = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def resolve_path(config: dict[str, Any], section: str, key: str) -> Path:
    return Path(config[section][key])
