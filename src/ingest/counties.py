"""Study-area counties and the clipping AOI.

Writes
    data/interim/study_area/counties.gpkg   county polygons, projected CRS
    data/interim/study_area/aoi.gpkg        county union + clip margin
"""
from __future__ import annotations

import urllib.request

import geopandas as gpd

from src.utils.config import load_params, path
from src.utils.log import get_logger

log = get_logger(__name__)


def download(params) -> str:
    url = params["sources"]["tiger_counties"]["url"]
    out = path("raw", "tiger", url.rsplit("/", 1)[-1])
    if not out.exists():
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, out)
    return str(out)


def select_counties(all_counties: gpd.GeoDataFrame, params) -> gpd.GeoDataFrame:
    want = params["study_area"]["counties"]
    c = all_counties.copy()
    c["county_fips"] = c["STATEFP"].astype(str) + c["COUNTYFP"].astype(str)
    c = c[c["county_fips"].isin(want)]
    missing = set(want) - set(c["county_fips"])
    if missing:
        raise ValueError(f"county FIPS not found in TIGER file: {sorted(missing)}")
    bad = {f: (want[f], n) for f, n in zip(c["county_fips"], c["NAME"])
           if want[f].lower() != str(n).lower()}
    if bad:
        raise ValueError(f"county names in params.yaml do not match TIGER: {bad}")
    return c[["county_fips", "NAME", "geometry"]].rename(columns={"NAME": "county_name"})


def main():
    params = load_params()
    crs = params["crs"]["projected"]
    counties = select_counties(gpd.read_file(download(params)), params).to_crs(crs)
    counties.to_file(path("interim", "study_area", "counties.gpkg"), driver="GPKG")
    margin = params["study_area"]["clip_margin_km"] * 1000
    max_buf = max(params["exposure"]["buffers_km"]) * 1000
    if margin < max_buf:
        raise ValueError("clip_margin_km must be at least the largest buffer")
    aoi = gpd.GeoDataFrame(geometry=[counties.union_all().buffer(margin)], crs=crs)
    aoi.to_file(path("interim", "study_area", "aoi.gpkg"), driver="GPKG")
    log.info("wrote %d counties and AOI", len(counties))


if __name__ == "__main__":
    main()
