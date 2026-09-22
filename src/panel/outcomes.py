"""School-year outcomes from the CCD enrollment pulls, MEPS and EDFacts.

Composition measures use harmonized race groups. Asian and Pacific Islander
are merged, and "two or more races" only exists from about 2009-10, so the
break-free diversity indices (entropy5, simpson5) use the five groups
reported in every year, renormalized to sum to one. entropy6/simpson6 add
the multiracial group and are NaN where it was not reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.urban_api import missing_to_nan, normalize_ncessch, read_cached
from src.utils.config import load_params
from src.utils.log import get_logger

log = get_logger(__name__)


def _total_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep all-sex, all-race rows unless the column is the one disaggregated."""
    for c in ("sex",):
        if c in df:
            df = df[pd.to_numeric(df[c], errors="coerce") == 99]
    return df


def grade_enrollment(params, ids=None) -> pd.DataFrame:
    """Wide table enr_<grade> per ncessch-year; enr_total is grade 99."""
    frames = []
    for g in params["sources"]["urban_api"]["enrollment_grades"]:
        try:
            d = read_cached(f"ccd_enrollment_grade/grade-{g}")
        except FileNotFoundError:
            log.warning("no cached enrollment for grade %s", g)
            continue
        d = _total_rows(d)
        if "race" in d:
            d = d[pd.to_numeric(d["race"], errors="coerce") == 99]
        d = missing_to_nan(d, ["enrollment"])
        d["ncessch"] = normalize_ncessch(d["ncessch"])
        if ids is not None:
            d = d[d["ncessch"].isin(ids)]
        col = "enr_total" if str(g) == "99" else f"enr_g{str(g).zfill(2)}"
        frames.append(d[["ncessch", "year"]].assign(**{col: d["enrollment"].to_numpy()}))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["ncessch", "year"], how="outer")
    return out


def race_enrollment(params, ids=None) -> pd.DataFrame:
    d = _total_rows(read_cached("ccd_enrollment_race"))
    d = missing_to_nan(d, ["enrollment", "race"])
    d["ncessch"] = normalize_ncessch(d["ncessch"])
    if ids is not None:
        d = d[d["ncessch"].isin(ids)]
    code_to_group = {c: g for g, codes in params["panel"]["race_groups"].items() for c in codes}
    d["group"] = d["race"].map(code_to_group)
    d = d.dropna(subset=["group"])
    wide = (d.pivot_table(index=["ncessch", "year"], columns="group", values="enrollment",
                          aggfunc=lambda v: v.sum(min_count=1))
            .add_prefix("enr_").reset_index())
    wide.columns.name = None
    return wide


def diversity(counts: pd.DataFrame) -> pd.DataFrame:
    """Shannon entropy (normalized to [0, 1] by log K) and Gini-Simpson index.

    Rows with any missing group or zero total get NaN.
    """
    c = counts.to_numpy(float)
    tot = c.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = c / tot
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(c.shape[1])
        simp = 1 - np.sum(p ** 2, axis=1)
    bad = np.isnan(c).any(axis=1) | (tot[:, 0] <= 0)
    ent[bad], simp[bad] = np.nan, np.nan
    return pd.DataFrame({"entropy": ent, "simpson": simp}, index=counts.index)


def composition(race: pd.DataFrame, params) -> pd.DataFrame:
    groups = list(params["panel"]["race_groups"])
    five = params["panel"]["consistent_race_groups"]
    out = race.copy()
    for g in groups:
        if f"enr_{g}" not in out:
            out[f"enr_{g}"] = np.nan
    all_cols = [f"enr_{g}" for g in groups]
    out["enr_race_total"] = out[all_cols].sum(axis=1, min_count=1)
    for g in groups:
        out[f"share_{g}"] = out[f"enr_{g}"] / out["enr_race_total"]
    d5 = diversity(out[[f"enr_{g}" for g in five]])
    d6 = diversity(out[all_cols])
    out["entropy5"], out["simpson5"] = d5["entropy"], d5["simpson"]
    out["entropy6"], out["simpson6"] = d6["entropy"], d6["simpson"]
    return out


def meps(ids=None) -> pd.DataFrame | None:
    try:
        d = read_cached("meps")
    except FileNotFoundError:
        return None
    d["ncessch"] = normalize_ncessch(d["ncessch"])
    if ids is not None:
        d = d[d["ncessch"].isin(ids)]
    keep = [c for c in d.columns if c.startswith("meps")]
    d = missing_to_nan(d, keep)
    return d[["ncessch", "year", *keep]]


def edfacts_standardized(params, ids=None) -> pd.DataFrame | None:
    """Proficiency midpoints standardized within year x grade x subject
    across the whole state, then averaged over grades 3-8 weighted by
    tested students. The state test changed in 2014-15 (CRCT to Milestones),
    so only within-year standardized values are comparable over time, and
    they describe relative position, not absolute achievement."""
    frames = []
    for g in params["sources"]["urban_api"]["edfacts_grades"]:
        try:
            d = read_cached(f"edfacts_assessments/grade-{g}")
        except FileNotFoundError:
            continue
        frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    for c in ("race", "sex", "lep", "homeless", "migrant", "disability",
              "econ_disadvantaged", "foster_care", "military_connected"):
        if c in d:
            d = d[pd.to_numeric(d[c], errors="coerce") == 99]
    d["ncessch"] = normalize_ncessch(d["ncessch"])
    d["grade_edfacts"] = pd.to_numeric(d["grade_edfacts"], errors="coerce")
    d = d[d["grade_edfacts"].between(3, 8)]
    long = []
    for subj in ("read", "math"):
        mid, n = f"{subj}_test_pct_prof_midpt", f"{subj}_test_num_valid"
        if mid not in d:
            continue
        s = missing_to_nan(d[["ncessch", "year", "grade_edfacts", mid, n]], [mid, n])
        s = s.rename(columns={mid: "pct_prof", n: "n_valid"}).assign(subject=subj)
        long.append(s)
    s = pd.concat(long, ignore_index=True).dropna(subset=["pct_prof"])
    grp = s.groupby(["year", "grade_edfacts", "subject"])["pct_prof"]
    s["z"] = (s["pct_prof"] - grp.transform("mean")) / grp.transform("std")
    s["w"] = s["n_valid"].fillna(1)
    s["zw"] = s["z"] * s["w"]
    agg = s.groupby(["ncessch", "year", "subject"]).agg(zw=("zw", "sum"), w=("w", "sum"),
                                                         n_valid=("n_valid", "sum"))
    agg["prof_z"] = agg["zw"] / agg["w"]
    wide = agg["prof_z"].unstack("subject").add_prefix("prof_z_")
    wide["n_tested"] = agg["n_valid"].groupby(level=[0, 1]).sum()
    wide = wide.reset_index()
    if ids is not None:
        wide = wide[wide["ncessch"].isin(ids)]
    return wide
