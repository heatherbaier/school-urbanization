"""Thin client for the Urban Institute Education Data Portal API.

Each (endpoint, year) request is written to data/raw/urban/<name>/<year>.parquet
and skipped on later runs, so an interrupted pull resumes where it stopped.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import load_params, path
from src.utils.log import get_logger

log = get_logger(__name__)

# Urban portal missing-data codes. Kept as-is in raw files, converted to
# NaN in interim files.
MISSING_CODES = (-1, -2, -3)


class UrbanClient:
    def __init__(self, params: dict | None = None):
        cfg = (params or load_params())["sources"]["urban_api"]
        self.base = cfg["base_url"].rstrip("/")
        self.timeout = cfg["timeout_s"]
        self.max_retries = cfg["max_retries"]
        self.sleep = cfg["sleep_between_requests_s"]
        self.session = requests.Session()

    def _get(self, url: str, query: dict | None = None) -> dict:
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=query, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                log.warning("HTTP %s for %s (attempt %d)", r.status_code, r.url, attempt + 1)
            except requests.RequestException as e:
                log.warning("%s for %s (attempt %d)", e, url, attempt + 1)
            time.sleep(2 ** attempt)
        raise RuntimeError(f"giving up on {url} after {self.max_retries} attempts")

    def fetch(self, endpoint: str, query: dict | None = None) -> pd.DataFrame:
        """Fetch every page of an endpoint, e.g. 'schools/ccd/directory/2015'."""
        url = f"{self.base}/{endpoint.strip('/')}/"
        rows: list[dict] = []
        payload = self._get(url, query)
        while True:
            rows.extend(payload.get("results", []))
            nxt = payload.get("next")
            if not nxt:
                break
            time.sleep(self.sleep)
            payload = self._get(nxt)
        expected = payload.get("count")
        if expected is not None and expected != len(rows):
            log.warning("%s returned %d rows, API reported %s", endpoint, len(rows), expected)
        return pd.DataFrame(rows)

    def fetch_cached(self, name: str, endpoint: str, year: int,
                     query: dict | None = None, overwrite: bool = False) -> Path | None:
        """Fetch one endpoint-year to data/raw/urban/<name>/<year>.parquet.

        Returns the path, or None when the endpoint has no data for that year.
        """
        out = path("raw", "urban", name, f"{year}.parquet")
        if out.exists() and not overwrite:
            return out
        try:
            df = self.fetch(endpoint, query)
        except FileNotFoundError:
            log.info("no data for %s %s", name, year)
            return None
        if df.empty:
            log.info("empty result for %s %s", name, year)
            return None
        df.to_parquet(out, index=False)
        log.info("wrote %s (%d rows)", out, len(df))
        return out


def read_cached(name: str, years=None) -> pd.DataFrame:
    """Concatenate every cached year for one pull."""
    d = path("raw", "urban", name, mkdir=False)
    files = sorted(d.glob("*.parquet"))
    if years is not None:
        keep = {str(y) for y in years}
        files = [f for f in files if f.stem in keep]
    if not files:
        raise FileNotFoundError(f"no cached files under {d}; run the ingest step first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def normalize_ncessch(s: pd.Series) -> pd.Series:
    """12-character zero-padded NCES school ID as a string."""
    s = s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    return s.str.zfill(12)


def missing_to_nan(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c].isin(MISSING_CODES), c] = pd.NA
    return df
