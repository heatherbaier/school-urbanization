"""Load config/params.yaml and resolve project paths."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "params.yaml"


@lru_cache(maxsize=None)
def load_params(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def path(key: str, *parts: str, params: dict | None = None, mkdir: bool = True) -> Path:
    """Absolute path under one of the configured data/output roots.

    path("raw", "ccd", "directory.parquet") -> <repo>/data/raw/ccd/directory.parquet
    The parent directory is created unless mkdir=False.
    """
    params = params or load_params()
    p = ROOT / params["paths"][key]
    for part in parts:
        p = p / part
    if mkdir:
        (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
    return p


def county_fips(params: dict | None = None) -> list[str]:
    params = params or load_params()
    return sorted(params["study_area"]["counties"])
