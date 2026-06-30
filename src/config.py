"""
Configuration loading utilities.

Loads `configs/config.yaml` into a nested, dot-accessible object so the
rest of the codebase never has to deal with raw dictionaries or hardcoded
hyperparameters. Every script and module pulls its settings from a single
`Config` instance, which keeps the paper's hyperparameters (Table 4, Eq.
1-27) in one auditable place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class DotDict(dict):
    """Dictionary subclass that also supports attribute-style access.

    Nested plain dicts are converted to DotDict *in place* (by swapping
    their class) the first time they're accessed, rather than being
    wrapped in a fresh copy. This means `cfg.data.raw_csv_path = "..."`
    mutates the same underlying dict that `cfg["data"]["raw_csv_path"]`
    would read from a copy would silently not persist otherwise.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[key] = value  # cache the converted object so future
            # accesses (and mutations through them) hit the same instance
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class Config(DotDict):
    """Top-level configuration object loaded from a YAML file."""

    @classmethod
    def load(cls, path: str | Path = "configs/config.yaml") -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found at '{path}'. Run scripts from the "
                f"repository root, or pass --config <path> explicitly."
            )
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)
        return cls(raw)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain (non-DotDict) nested dictionary copy."""

        def _plain(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _plain(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_plain(v) for v in obj]
            return obj

        return _plain(self)


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    """Convenience wrapper used throughout the scripts."""
    return Config.load(path)
