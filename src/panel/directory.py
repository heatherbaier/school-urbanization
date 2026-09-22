"""Clean the CCD directory into a school-year table for the study area.

Output: data/interim/ccd/directory.parquet, one row per ncessch x year.
A school is in the study set if, in any year, its CCD county code is a
study county or its point falls inside a study county. All of its years
are then kept, including years before opening or after closing, so the
panel never conditions on staying in the area.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from src.ingest.urban_api import missing_to_nan, normalize_ncessch, read_cached
from src.utils.config import county_fips, load_params, path
from src.utils.geo import points_gdf
from src.utils.log import get_logger

log = get_logger(__name__)

KEEP = [
    "ncessch", "year", "school_name", "leaid", "lea_name", "state_leaid", "seasch",
    "county_code", "latitude", "longitude", "school_type", "school_level",
    "school_status", "charter", "magnet", "virtual", "lowest_grade_offered",
    "highest_grade_offered", "urban_centric_locale", "title_i_eligible",
    "enrollment", "teachers_fte", "free_lunch", "reduced_price_lunch",
    "free_or_reduced_price_lunch", "direct_certification",
]
NUMERIC = [
    "school_type", "school_level", "school_status", "charter", "magnet", "virtual",
    "lowest_grade_offered", "highest_grade_offered", "urban_centric_locale",
    "title_i_eligible", "enrollment", "teachers_fte", "free_lunch",
    "reduced_price_lunch", "free_or_reduced_price_lunch", "direct_certification",
]


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[[c for c in KEEP if c in raw.columns]].copy()
    df["ncessch"] = normalize_ncessch(df["ncessch"])
    df["year"] = df["year"].astype(int)
    df["leaid"] = df["leaid"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(7)
    df["county_code"] = (pd.to_numeric(df["county_code"], errors="coerce")
                         .astype("Int64").astype("string").str.zfill(5))
    df = missing_to_nan(df, NUMERIC)
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = (df["latitude"].abs() < 1) | (df["longitude"].abs() < 1)
    df.loc[bad, ["latitude", "longitude"]] = np.nan
    dup = df.duplicated(["ncessch", "year"])
    if dup.any():
        log.warning("dropping %d duplicate ncessch-year rows", dup.sum())
        df = df[~dup]
    return df.sort_values(["ncessch", "year"]).reset_index(drop=True)


def study_ids(df: pd.DataFrame, params) -> set[str]:
    fips = set(county_fips(params))
    by_code = set(df.loc[df["county_code"].isin(fips), "ncessch"])
    counties_fp = path("interim", "study_area", "counties.gpkg", mkdir=False)
    by_point: set[str] = set()
    if counties_fp.exists():
        counties = gpd.read_file(counties_fp).to_crs("EPSG:4326")
        pts = points_gdf(df.dropna(subset=["latitude", "longitude"]))
        hit = gpd.sjoin(pts, counties[["geometry"]], predicate="within", how="inner")
        by_point = set(hit["ncessch"])
    else:
        log.warning("counties.gpkg missing; selecting on CCD county code only. "
                    "Run python -m src.ingest.counties first.")
    log.info("study schools: %d by county code, %d by point, %d union",
             len(by_code), len(by_point), len(by_code | by_point))
    return by_code | by_point


def flag_primary(df: pd.DataFrame, params) -> pd.DataFrame:
    cfg = params["panel"]
    df = df.copy()
    df["is_charter"] = df["charter"].eq(1)
    df["is_virtual"] = df["virtual"].eq(1) if "virtual" in df else False
    primary = df["school_type"].isin(cfg["primary_school_types"])
    if cfg["exclude_charter_from_primary"]:
        primary &= ~df["is_charter"]
    if cfg["exclude_virtual_from_primary"]:
        primary &= ~df["is_virtual"]
    df["in_primary_year"] = primary
    # A school is in the primary sample if it qualifies in most of its years,
    # so a one-year coding blip does not move it in and out of the sample.
    df["in_primary"] = df.groupby("ncessch")["in_primary_year"].transform("mean") > 0.5
    return df


def main():
    params = load_params()
    y0, y1 = params["years"]["school_start"], params["years"]["school_end"]
    df = clean(read_cached("ccd_directory", years=range(y0, y1 + 1)))
    df = df[df["ncessch"].isin(study_ids(df, params))]
    df = flag_primary(df, params)
    out = path("interim", "ccd", "directory.parquet")
    df.to_parquet(out, index=False)
    log.info("wrote %s: %d school-years, %d schools", out, len(df), df["ncessch"].nunique())


if __name__ == "__main__":
    main()
