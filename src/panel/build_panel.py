"""Assemble the school-year analysis panel.

One row per ncessch x year for every study-area school, including closed
and newly opened ones (openings and closures are outcomes, so dropping
them would condition on treatment). Exposure columns are attached when the
exposure steps have been run, and left out otherwise.

Output: data/processed/school_year_panel.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.panel import outcomes
from src.utils.config import load_params, path
from src.utils.log import get_logger

log = get_logger(__name__)


def attach_impervious(panel: pd.DataFrame) -> pd.DataFrame:
    fp = path("interim", "exposure", "impervious_buffers.parquet", mkdir=False)
    if not fp.exists():
        log.info("no impervious extraction yet; skipping")
        return panel
    imp = pd.read_parquet(fp)
    wide = imp.pivot_table(index=["site_id", "year"], columns="buffer_km", values="imp_mean")
    wide.columns = [f"imp_{int(b)}km" for b in wide.columns]
    return panel.merge(wide.reset_index(), on=["site_id", "year"], how="left")


def attach_smod(panel: pd.DataFrame) -> pd.DataFrame:
    """Each school year takes the most recent epoch at or before it."""
    fp = path("interim", "exposure", "smod_sites.parquet", mkdir=False)
    if not fp.exists():
        log.info("no GHS-SMOD extraction yet; skipping")
        return panel
    smod = pd.read_parquet(fp)
    epochs = np.sort(smod["epoch"].unique())
    idx = np.searchsorted(epochs, panel["year"].to_numpy(), side="right") - 1
    panel = panel.assign(smod_epoch=np.where(idx >= 0, epochs[np.clip(idx, 0, None)], np.nan))
    smod = smod.rename(columns={"epoch": "smod_epoch", "projected": "smod_projected"})
    return panel.merge(smod, on=["site_id", "smod_epoch"], how="left")


def lifecycle(panel: pd.DataFrame, params) -> pd.DataFrame:
    y0, y1 = params["years"]["school_start"], params["years"]["school_end"]
    g = panel.groupby("school_uid")["year"]
    panel["uid_first_year"] = g.transform("min")
    panel["uid_last_year"] = g.transform("max")
    # Left-censored schools already existed at the start of the data.
    panel["opened_in_window"] = panel["uid_first_year"] > y0
    panel["closed_in_window"] = panel["uid_last_year"] < y1
    panel["is_opening_year"] = panel["opened_in_window"] & panel["year"].eq(panel["uid_first_year"])
    panel["is_closing_year"] = panel["closed_in_window"] & panel["year"].eq(panel["uid_last_year"])
    return panel


def derived(panel: pd.DataFrame) -> pd.DataFrame:
    enr = panel["enr_total"].where(panel["enr_total"].notna(), panel.get("enrollment"))
    panel["enr"] = enr
    panel["log_enr"] = np.log(enr.where(enr > 0))
    panel = panel.sort_values(["school_uid", "year"])
    prev = panel.groupby("school_uid")["log_enr"].shift()
    gap = panel["year"] - panel.groupby("school_uid")["year"].shift()
    panel["enr_growth"] = np.where(gap == 1, panel["log_enr"] - prev, np.nan)
    panel["student_teacher_ratio"] = enr / panel["teachers_fte"].where(panel["teachers_fte"] > 0)
    # Free/reduced lunch is not comparable after the Community Eligibility
    # Provision (from 2014-15 in Georgia); kept for descriptives and early years.
    panel["frl_share"] = panel["free_or_reduced_price_lunch"] / enr.where(enr > 0)
    return panel


def main():
    params = load_params()
    d = pd.read_parquet(path("interim", "ccd", "directory.parquet"))
    sites = pd.read_parquet(path("interim", "ccd", "school_sites.parquet"))
    panel = d.merge(sites, on=["ncessch", "year"], how="left", validate="one_to_one")
    ids = set(panel["ncessch"])

    panel = panel.merge(outcomes.grade_enrollment(params, ids), on=["ncessch", "year"], how="left")
    comp = outcomes.composition(outcomes.race_enrollment(params, ids), params)
    panel = panel.merge(comp, on=["ncessch", "year"], how="left")
    for extra in (outcomes.meps(ids), outcomes.edfacts_standardized(params, ids)):
        if extra is not None:
            panel = panel.merge(extra, on=["ncessch", "year"], how="left")

    panel = derived(lifecycle(panel, params))
    panel = attach_smod(attach_impervious(panel))

    fp = path("processed", "school_year_panel.parquet")
    panel.to_parquet(fp, index=False)
    log.info("wrote %s: %d rows, %d schools, %d sites", fp, len(panel),
             panel["school_uid"].nunique(), panel["site_id"].nunique())


if __name__ == "__main__":
    main()
