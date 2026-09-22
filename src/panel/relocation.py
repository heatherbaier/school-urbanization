"""Split each school's history into sites and flag relocations.

A relocated school is a different exposure, so exposure is extracted per
site, not per school. For each school_uid we walk its yearly coordinates in
order. A coordinate more than move_threshold_m from the current site anchor
is a candidate move. If a later coordinate returns within return_tolerance_m
of the anchor inside return_window_years, the jump is treated as geocode
noise (coord_outlier) instead of a move. Otherwise a new site starts.

Outputs
    data/interim/ccd/school_sites.parquet   one row per ncessch x year with
        school_uid, site_id, coord_outlier, coord_imputed, moved_this_year,
        move_distance_m, move_needs_review
    data/interim/ccd/sites.parquet          one row per site with the median
        coordinate over its non-outlier years (used for exposure)
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from src.utils.config import load_params, path
from src.utils.geo import projected_xy
from src.utils.log import get_logger

log = get_logger(__name__)


def assign_sites(xy: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xy is an (n, 2) array of one school's coordinates in year order, NaN
    rows allowed. Returns site index, outlier flag, and distance of the move
    that starts a new site (NaN elsewhere)."""
    n = len(xy)
    site = np.full(n, -1)
    outlier = np.zeros(n, bool)
    move_d = np.full(n, np.nan)
    anchor = None
    k = 0
    for i in range(n):
        if np.isnan(xy[i]).any():
            continue
        if anchor is None:
            anchor, site[i] = xy[i], k
            continue
        d = float(np.hypot(*(xy[i] - anchor)))
        if d <= cfg["move_threshold_m"]:
            site[i] = k
            continue
        ahead = [j for j in range(i + 1, min(n, i + 1 + cfg["return_window_years"]))
                 if not np.isnan(xy[j]).any()]
        returns = any(np.hypot(*(xy[j] - anchor)) <= cfg["return_tolerance_m"] for j in ahead)
        if returns:
            site[i], outlier[i] = k, True
        else:
            k += 1
            anchor, site[i], move_d[i] = xy[i], k, d
    return site, outlier, move_d


def build(df: pd.DataFrame, uid: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = params["relocation"]
    d = df.merge(uid, on="ncessch", how="left")
    if "x" not in d:
        d = projected_xy(d, params["crs"]["projected"])
    d = d.sort_values(["school_uid", "year", "ncessch"]).reset_index(drop=True)

    # Two IDs of one school_uid never overlap in time (links run forward),
    # so one row per school_uid-year remains.
    parts = []
    for u, g in d.groupby("school_uid", sort=False):
        site, outl, move_d = assign_sites(g[["x", "y"]].to_numpy(float), cfg)
        parts.append(pd.DataFrame({"idx": g.index, "site_idx": site,
                                   "coord_outlier": outl, "move_distance_m": move_d}))
    s = pd.concat(parts).set_index("idx")
    d = d.join(s)

    # Years with no coordinate inherit the surrounding site.
    d["coord_imputed"] = d["site_idx"].eq(-1)
    d["site_idx"] = d["site_idx"].replace(-1, np.nan)
    d["site_idx"] = d.groupby("school_uid")["site_idx"].transform(lambda v: v.ffill().bfill())
    no_coords = d["site_idx"].isna()
    if no_coords.any():
        log.warning("%d school-years belong to schools with no coordinates at all",
                    d.loc[no_coords, "school_uid"].nunique())
    d["site_id"] = np.where(no_coords, pd.NA,
                            d["school_uid"] + "_" + d["site_idx"].fillna(0).astype(int).astype(str))
    d["moved_this_year"] = d["move_distance_m"].notna()
    d["move_needs_review"] = d["moved_this_year"] & (d["move_distance_m"] < cfg["review_threshold_m"])

    good = d[~d["coord_outlier"] & ~d["coord_imputed"] & d["site_id"].notna()]
    sites = good.groupby("site_id").agg(
        school_uid=("school_uid", "first"),
        x=("x", "median"), y=("y", "median"),
        first_year=("year", "min"), last_year=("year", "max"),
        n_years=("year", "size"),
    ).reset_index()
    # Spread of the yearly coordinates around the site median, a geocode
    # quality check.
    dev = good.merge(sites[["site_id", "x", "y"]], on="site_id", suffixes=("", "_med"))
    dev["dev"] = np.hypot(dev["x"] - dev["x_med"], dev["y"] - dev["y_med"])
    sites["max_dev_m"] = sites["site_id"].map(dev.groupby("site_id")["dev"].max())
    ll = gpd.GeoSeries(gpd.points_from_xy(sites["x"], sites["y"]),
                       crs=params["crs"]["projected"]).to_crs("EPSG:4326")
    sites["longitude"], sites["latitude"] = ll.x.to_numpy(), ll.y.to_numpy()

    cols = ["ncessch", "year", "school_uid", "site_id", "coord_outlier", "coord_imputed",
            "moved_this_year", "move_distance_m", "move_needs_review"]
    return d[cols], sites


def main():
    params = load_params()
    df = pd.read_parquet(path("interim", "ccd", "directory.parquet"))
    uid = pd.read_parquet(path("interim", "ccd", "school_uid.parquet"))
    school_sites, sites = build(df, uid, params)
    school_sites.to_parquet(path("interim", "ccd", "school_sites.parquet"), index=False)
    sites.to_parquet(path("interim", "ccd", "sites.parquet"), index=False)
    log.info("%d schools, %d sites, %d relocations (%d need review), %d outlier coordinates",
             school_sites["school_uid"].nunique(), len(sites),
             int(school_sites["moved_this_year"].sum()),
             int(school_sites["move_needs_review"].sum()),
             int(school_sites["coord_outlier"].sum()))


if __name__ == "__main__":
    main()
