"""Clip Annual NLCD fractional impervious surface to the study AOI.

Reads each year's CONUS raster (remote COG via /vsicurl/ or a local file,
per sources.nlcd.fctimp_template) with a window covering the AOI, and writes
data/interim/nlcd/fctimp_<year>.tif in the native EPSG:5070 grid. Rasters
are never resampled. Every year shares one grid, which buffers.py checks.

    python -m src.exposure.nlcd [--years 2001 2002 ...]
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds

from src.utils.config import load_params, path
from src.utils.geo import snap_window
from src.utils.log import get_logger

log = get_logger(__name__)


def aoi_bounds(crs: str) -> tuple[float, float, float, float]:
    aoi = gpd.read_file(path("interim", "study_area", "aoi.gpkg", mkdir=False))
    return tuple(aoi.to_crs(crs).total_bounds)


def clip_year(src_path: str, bounds, out_path, nodata_default: int) -> None:
    with rasterio.open(src_path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        win = snap_window(win)
        data = src.read(1, window=win, boundless=True,
                        fill_value=src.nodata if src.nodata is not None else nodata_default)
        prof = src.profile.copy()
        prof.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                    transform=src.window_transform(win), compress="deflate",
                    tiled=True, blockxsize=512, blockysize=512,
                    nodata=src.nodata if src.nodata is not None else nodata_default)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(data, 1)


def main(argv=None):
    params = load_params()
    cfg = params["sources"]["nlcd"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", type=int,
                    default=list(range(params["years"]["land_start"], params["years"]["land_end"] + 1)))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    bounds = aoi_bounds(params["crs"]["nlcd"])
    for year in args.years:
        out = path("interim", "nlcd", f"fctimp_{year}.tif")
        if out.exists() and not args.overwrite:
            continue
        src = cfg["fctimp_template"].format(year=year)
        try:
            clip_year(src, bounds, out, cfg["nodata"])
            log.info("wrote %s", out)
        except rasterio.errors.RasterioIOError as e:
            log.error("could not read %s (%s)", src, e)


if __name__ == "__main__":
    main()
