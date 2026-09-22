"""Fringe sample and treatment timing for each school.

All thresholds come from params["definitions"]. The functions take the
definitions as arguments so the diagnostics sensitivity grid can call them
with other values.

Per school (school_uid) this builds
    baseline_year        first panel year (mode first_year) or the fixed year
    imp_base             impervious fraction in the fringe buffer at baseline
    km_to_cluster_base   distance to an urban cluster at the baseline epoch
    is_fringe            low impervious and near an urban cluster at baseline
    g_land               first year the event buffer reaches the event threshold
                         and stays there for persistence_years (NaN if none).
                         The threshold is imp_base_event + change in change
                         mode, or the fixed threshold in level mode
    land_censored        crossed near the end of the land series, too late to
                         confirm persistence, so neither treated nor never-treated
    g_pop                first observed GHS-SMOD epoch after baseline in which
                         the school's cell moves from a rural to an urban class
    n_pre_years          panel years with enrollment between baseline and g_land-1
    group                treated | never_treated | censored | not_fringe

Exposure follows the school's site. Years before the school's first panel
year use its first site, and years after its last panel year use its last
site, so a school has a land series before it opens.

    python -m src.exposure.events   -> data/processed/treatment.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_params, path
from src.utils.log import get_logger

log = get_logger(__name__)


def site_by_year(panel: pd.DataFrame, years) -> pd.DataFrame:
    """school_uid x year -> site_id over the full land-year range."""
    p = (panel.dropna(subset=["site_id"])
         .sort_values(["school_uid", "year"])
         .drop_duplicates(["school_uid", "year"]))
    grid = pd.MultiIndex.from_product([p["school_uid"].unique(), list(years)],
                                      names=["school_uid", "year"])
    s = p.set_index(["school_uid", "year"])["site_id"].reindex(grid)
    s = s.groupby(level=0).transform(lambda v: v.ffill().bfill())
    return s.reset_index()


def exposure_series(sites: pd.DataFrame, imp: pd.DataFrame, buffer_km) -> pd.DataFrame:
    """school_uid x year impervious fraction for one buffer."""
    i = imp.loc[imp["buffer_km"] == buffer_km, ["site_id", "year", "imp_mean"]]
    return sites.merge(i, on=["site_id", "year"], how="left")


def land_event(years: np.ndarray, values: np.ndarray, after_year: int,
               threshold: float, persistence: int) -> tuple[float, bool]:
    """First year > after_year that is at or above threshold for `persistence`
    consecutive observed years. Returns (event_year or NaN, censored).

    censored is True when the only qualifying run starts too close to the end
    of the series to be confirmed.
    """
    order = np.argsort(years)
    years, values = np.asarray(years)[order], np.asarray(values, float)[order]
    above = values >= threshold          # NaN compares False
    last = years.max() if len(years) else None
    for k in np.nonzero((years > after_year) & above)[0]:
        run = years[k:k + persistence]
        consecutive = len(run) == persistence and np.all(np.diff(run) == 1)
        if consecutive and above[k:k + persistence].all():
            return float(years[k]), False
        tail = above[k:]
        if years[k] + persistence - 1 > last and tail.all() and np.all(np.diff(years[k:]) == 1):
            return np.nan, True
    return np.nan, False


def pop_event(epochs: np.ndarray, codes: np.ndarray, baseline_year: int,
              from_codes, to_codes) -> float:
    """First epoch after the baseline epoch whose class is urban, given a rural
    class at the baseline epoch (the latest epoch at or before baseline)."""
    order = np.argsort(epochs)
    epochs, codes = np.asarray(epochs)[order], np.asarray(codes)[order]
    base = np.nonzero(epochs <= baseline_year)[0]
    if not len(base) or codes[base[-1]] not in from_codes:
        return np.nan
    later = np.nonzero((epochs > epochs[base[-1]]) & np.isin(codes, to_codes))[0]
    return float(epochs[later[0]]) if len(later) else np.nan


def baselines(panel: pd.DataFrame, params: dict, defs: dict) -> pd.DataFrame:
    y0 = params["years"]["school_start"]
    g = panel.groupby("school_uid")
    b = pd.DataFrame({
        "first_year": g["year"].min(),
        "last_year": g["year"].max(),
        "in_primary": g["in_primary"].max() if "in_primary" in panel else True,
    })
    if defs["baseline"]["mode"] == "fixed_year":
        fy = defs["baseline"]["fixed_year"]
        open_then = panel.loc[panel["year"] == fy, "school_uid"].unique()
        b["baseline_year"] = np.where(b.index.isin(open_then), fy, np.nan)
    else:
        b["baseline_year"] = b["first_year"].clip(lower=y0)
    return b.reset_index()


def build(panel: pd.DataFrame, imp: pd.DataFrame, smod: pd.DataFrame | None,
          params: dict, defs: dict | None = None) -> pd.DataFrame:
    defs = defs or params["definitions"]
    land_years = sorted(imp["year"].unique())
    sites = site_by_year(panel, land_years)
    out = baselines(panel, params, defs)

    # Impervious at baseline in the fringe buffer.
    fr = defs["fringe"]
    ser_f = exposure_series(sites, imp, fr["buffer_km"])
    base_imp = ser_f.merge(out[["school_uid", "baseline_year"]], on="school_uid")
    base_imp = base_imp[base_imp["year"] == base_imp["baseline_year"]]
    out = out.merge(base_imp[["school_uid", "imp_mean"]].rename(columns={"imp_mean": "imp_base"}),
                    on="school_uid", how="left")

    # GHS-SMOD at the baseline epoch, and the population-based event.
    out["km_to_cluster_base"] = np.nan
    out["smod_base"] = np.nan
    out["g_pop"] = np.nan
    if smod is not None:
        obs = smod[~smod["projected"]] if "projected" in smod else smod
        epochs = np.sort(obs["epoch"].unique())
        ep_sites = site_by_year(panel, epochs)
        sm = ep_sites.rename(columns={"year": "epoch"}).merge(obs, on=["site_id", "epoch"], how="left")
        sm_by = {u: g for u, g in sm.groupby("school_uid")}
        pe = defs["pop_event"]
        kms, codes, gpop = [], [], []
        for u, by in zip(out["school_uid"], out["baseline_year"]):
            g = sm_by.get(u)
            if g is None or np.isnan(by):
                kms.append(np.nan), codes.append(np.nan), gpop.append(np.nan)
                continue
            before = g[g["epoch"] <= by].dropna(subset=["smod_code"])
            if before.empty:
                kms.append(np.nan), codes.append(np.nan)
            else:
                r = before.iloc[-1]
                kms.append(r["km_to_urban_cluster"]), codes.append(r["smod_code"])
            gg = g.dropna(subset=["smod_code"])
            gpop.append(pop_event(gg["epoch"].to_numpy(), gg["smod_code"].to_numpy(), by,
                                  pe["from_codes"], pe["to_codes"]))
        out["km_to_cluster_base"], out["smod_base"], out["g_pop"] = kms, codes, gpop

    near = out["km_to_cluster_base"] <= fr["max_km_to_urban_cluster"]
    if smod is None:
        near = pd.Series(True, index=out.index)
    out["is_fringe"] = (out["imp_base"] < fr["max_impervious"]) & near & out["baseline_year"].notna()

    # Land-based event.
    le = defs["land_event"]
    mode = le.get("mode", "level")
    ser_e = exposure_series(sites, imp, le["buffer_km"])
    base_e = ser_e.merge(out[["school_uid", "baseline_year"]], on="school_uid")
    base_e = base_e[base_e["year"] == base_e["baseline_year"]].set_index("school_uid")["imp_mean"]
    out["imp_base_event"] = out["school_uid"].map(base_e)
    if mode == "change":
        out["event_threshold"] = out["imp_base_event"] + le["change"]
    elif mode == "level":
        out["event_threshold"] = float(le["threshold"])
    else:
        raise ValueError(f"land_event.mode must be 'change' or 'level', got {mode!r}")
    ser_by = {u: g for u, g in ser_e.groupby("school_uid")}
    g_land, cens = [], []
    for u, by, thr in zip(out["school_uid"], out["baseline_year"], out["event_threshold"]):
        g = ser_by.get(u)
        if g is None or np.isnan(by) or np.isnan(thr):
            g_land.append(np.nan), cens.append(False)
            continue
        e, c = land_event(g["year"].to_numpy(), g["imp_mean"].to_numpy(), int(by),
                          thr, le["persistence_years"])
        g_land.append(e), cens.append(c)
    out["g_land"], out["land_censored"] = g_land, cens

    # Pre and post years with an enrollment observation.
    obs = panel.loc[panel["enr"].notna(), ["school_uid", "year"]] if "enr" in panel else panel[["school_uid", "year"]]
    o = obs.merge(out[["school_uid", "baseline_year", "g_land"]], on="school_uid")
    pre = o[(o["year"] >= o["baseline_year"]) & (o["year"] < o["g_land"])].groupby("school_uid").size()
    post = o[o["year"] >= o["g_land"]].groupby("school_uid").size()
    out["n_pre_years"] = out["school_uid"].map(pre).fillna(0).astype(int)
    out["n_post_years"] = out["school_uid"].map(post).fillna(0).astype(int)

    out["group"] = np.select(
        [~out["is_fringe"], out["g_land"].notna(), out["land_censored"]],
        ["not_fringe", "treated", "censored"], default="never_treated")
    return out


def load_inputs():
    panel = pd.read_parquet(path("processed", "school_year_panel.parquet"))
    imp = pd.read_parquet(path("interim", "exposure", "impervious_buffers.parquet"))
    fp = path("interim", "exposure", "smod_sites.parquet", mkdir=False)
    smod = pd.read_parquet(fp) if fp.exists() else None
    if smod is None:
        log.warning("no GHS-SMOD extraction; fringe ignores distance to urban clusters")
    return panel, imp, smod


def main():
    params = load_params()
    panel, imp, smod = load_inputs()
    t = build(panel, imp, smod, params)
    fp = path("processed", "treatment.parquet")
    t.to_parquet(fp, index=False)
    log.info("wrote %s: %s", fp, t["group"].value_counts().to_dict())


if __name__ == "__main__":
    main()
