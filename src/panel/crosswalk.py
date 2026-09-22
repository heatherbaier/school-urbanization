"""Link NCES IDs that belong to the same school and assign a stable school_uid.

An NCES ID can change when a school moves between districts or a state
re-issues codes. We link an ID that ends to one that starts shortly after
when the two sit at nearly the same place and either have similar names or
share a state school ID. Links are one-to-one (closest first); ambiguous
cases are kept in the crosswalk with a flag for manual review.

Output: data/interim/ccd/id_crosswalk.parquet with columns
    old_ncessch, new_ncessch, old_last_year, new_first_year, distance_m,
    name_similarity, same_state_id, n_candidates, match_basis
and data/interim/ccd/school_uid.parquet mapping ncessch -> school_uid.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from src.utils.config import load_params, path
from src.utils.geo import dist, projected_xy
from src.utils.log import get_logger

log = get_logger(__name__)

_ABBREV = [
    (r"\belem(entary)?\b|\bes\b", "elementary"),
    (r"\bmid(dle)?\b|\bms\b", "middle"),
    (r"\bhigh\b|\bhs\b", "high"),
    (r"\bacad(emy)?\b", "academy"),
    (r"\bsch(ool)?\b", ""),
    (r"\bchs\b", "charter high"),
]


def norm_name(name) -> str:
    s = str(name or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    for pat, rep in _ABBREV:
        s = re.sub(pat, rep, s)
    return re.sub(r"\s+", " ", s).strip()


def name_similarity(a, b) -> float:
    return SequenceMatcher(None, norm_name(a), norm_name(b)).ratio()


def id_spans(df: pd.DataFrame) -> pd.DataFrame:
    """First/last year, name, state ID and location at each end of an ID's span."""
    d = df.sort_values(["ncessch", "year"])
    has_xy = d.dropna(subset=["x", "y"])
    g = d.groupby("ncessch")
    gx = has_xy.groupby("ncessch")
    spans = pd.DataFrame({
        "first_year": g["year"].min(),
        "last_year": g["year"].max(),
        "first_name": g["school_name"].first(),
        "last_name": g["school_name"].last(),
        "first_seasch": g["seasch"].first() if "seasch" in d else None,
        "last_seasch": g["seasch"].last() if "seasch" in d else None,
        "first_x": gx["x"].first(), "first_y": gx["y"].first(),
        "last_x": gx["x"].last(), "last_y": gx["y"].last(),
    })
    return spans.reset_index()


def candidate_links(spans: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ends = spans.rename(columns=lambda c: f"old_{c}")
    starts = spans.rename(columns=lambda c: f"new_{c}")
    pairs = []
    for gap in range(0, cfg["max_gap_years"] + 1):
        e = ends.assign(_k=ends["old_last_year"] + 1 + gap)
        s = starts.assign(_k=starts["new_first_year"])
        pairs.append(e.merge(s, on="_k"))
    c = pd.concat(pairs, ignore_index=True).drop(columns="_k")
    c = c[c["old_ncessch"] != c["new_ncessch"]]
    c["distance_m"] = dist(c["old_last_x"], c["old_last_y"], c["new_first_x"], c["new_first_y"])
    c = c[c["distance_m"] <= cfg["max_distance_m"]].copy()
    if c.empty:
        return c
    c["name_similarity"] = [name_similarity(a, b) for a, b in zip(c["old_last_name"], c["new_first_name"])]
    c["same_state_id"] = (c["old_last_seasch"].notna()
                          & (c["old_last_seasch"].astype(str) == c["new_first_seasch"].astype(str)))
    by_name = c["name_similarity"] >= cfg["min_name_similarity"]
    by_sid = c["same_state_id"] & bool(cfg.get("use_state_school_id", True))
    c = c[by_name | by_sid].copy()
    c["match_basis"] = np.where(by_name[c.index] & by_sid[c.index], "name+state_id",
                                np.where(by_name[c.index], "name", "state_id"))
    return c


def resolve_one_to_one(c: pd.DataFrame) -> pd.DataFrame:
    """Greedy one-to-one matching, closest and most similar first."""
    if c.empty:
        return c.assign(n_candidates=pd.Series(dtype=int))
    c = c.copy()
    n_old = c.groupby("old_ncessch")["new_ncessch"].transform("size")
    n_new = c.groupby("new_ncessch")["old_ncessch"].transform("size")
    c["n_candidates"] = np.maximum(n_old, n_new)
    c = c.sort_values(["distance_m", "name_similarity"], ascending=[True, False])
    used_old, used_new, keep = set(), set(), []
    for i, r in c.iterrows():
        if r["old_ncessch"] in used_old or r["new_ncessch"] in used_new:
            continue
        used_old.add(r["old_ncessch"])
        used_new.add(r["new_ncessch"])
        keep.append(i)
    return c.loc[keep]


def assign_uid(ids, links: pd.DataFrame) -> pd.DataFrame:
    """Follow links forward; school_uid is the earliest ID in each chain."""
    nxt = dict(zip(links["old_ncessch"], links["new_ncessch"]))
    has_prev = set(links["new_ncessch"])
    uid = {}
    for start in ids:
        if start in has_prev:
            continue
        cur = start
        while cur is not None and cur not in uid:
            uid[cur] = start
            cur = nxt.get(cur)
    for i in ids:  # defensive, cycles cannot occur because links go forward in time
        uid.setdefault(i, i)
    return pd.DataFrame({"ncessch": list(uid), "school_uid": list(uid.values())})


def build(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = params["crosswalk"]
    if "x" not in df:
        df = projected_xy(df, params["crs"]["projected"])
    spans = id_spans(df)
    links = resolve_one_to_one(candidate_links(spans, cfg))
    cols = ["old_ncessch", "new_ncessch", "old_last_year", "new_first_year",
            "old_last_name", "new_first_name", "distance_m", "name_similarity",
            "same_state_id", "n_candidates", "match_basis"]
    links = links[[c for c in cols if c in links]].reset_index(drop=True)
    uid = assign_uid(sorted(df["ncessch"].unique()), links)
    return links, uid


def main():
    params = load_params()
    df = pd.read_parquet(path("interim", "ccd", "directory.parquet"))
    links, uid = build(df, params)
    links.to_parquet(path("interim", "ccd", "id_crosswalk.parquet"), index=False)
    uid.to_parquet(path("interim", "ccd", "school_uid.parquet"), index=False)
    log.info("%d ID links (%d ambiguous), %d IDs -> %d school_uids",
             len(links), int((links.get("n_candidates", pd.Series(dtype=int)) > 1).sum()),
             len(uid), uid["school_uid"].nunique())


if __name__ == "__main__":
    main()
