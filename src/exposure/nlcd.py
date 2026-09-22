"""Clip Annual NLCD fractional impervious surface to the study AOI.

Reads each year's CONUS raster with a window covering the AOI and writes
data/interim/nlcd/fctimp_<year>.tif in the native EPSG:5070 grid. Rasters
are never resampled. Every year shares one grid, which buffers.py checks.

The source is set by sources.nlcd.fctimp_template (see params.yaml).

    python -m src.exposure.nlcd --probe          # which source URLs answer
    python -m src.exposure.nlcd [--years 2001 2002 ...]
"""
from __future__ import annotations

import argparse
import shutil
import zipfile

import geopandas as gpd
import rasterio
import requests
from rasterio.windows import from_bounds

from src.utils.config import load_params, path
from src.utils.geo import snap_window
from src.utils.log import get_logger

log = get_logger(__name__)


def aoi_bounds(crs: str) -> tuple[float, float, float, float]:
    aoi = gpd.read_file(path("interim", "study_area", "aoi.gpkg", mkdir=False))
    return tuple(aoi.to_crs(crs).total_bounds)


def url_exists(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=60)
        return r.status_code == 200
    except requests.RequestException:
        return False


def probe(cfg: dict, year: int) -> list[tuple[str, bool]]:
    return [(c, url_exists(c.format(year=year))) for c in cfg["fctimp_candidates"]]


def resolve_template(cfg: dict, year: int) -> str:
    t = cfg["fctimp_template"]
    if t != "auto":
        return t
    for cand, ok in probe(cfg, year):
        if ok:
            log.info("using NLCD source pattern %s", cand)
            return cand
    raise RuntimeError(
        "none of sources.nlcd.fctimp_candidates answered for year "
        f"{year}. Find the current file URL on mrlc.gov (Data > Annual NLCD) "
        "and set sources.nlcd.fctimp_template to it, with {year} in place of the year.")


def download(url: str) -> str:
    out = path("raw", "nlcd", url.rsplit("/", 1)[-1])
    if not out.exists():
        log.info("downloading %s", url)
        tmp = out.with_suffix(out.suffix + ".part")
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                shutil.copyfileobj(r.raw, f, length=16 << 20)
        tmp.rename(out)
    return str(out)


def open_source(src: str) -> tuple[str, str | None]:
    """Return a GDAL-readable path, plus a local zip to delete afterwards."""
    if src.lower().endswith(".zip"):
        zp = download(src) if src.startswith("http") else src
        with zipfile.ZipFile(zp) as z:
            tif = next(n for n in z.namelist() if n.lower().endswith(".tif")
                       and not n.lower().endswith(".aux.tif"))
        return f"/vsizip/{zp}/{tif}", (zp if src.startswith("http") else None)
    if src.startswith("http"):
        return f"/vsicurl/{src}", None
    return src, None


def clip_year(src_path: str, bounds, out_path, nodata_default: int) -> None:
    with rasterio.open(src_path) as src:
        win = snap_window(from_bounds(*bounds, transform=src.transform))
        nod = src.nodata if src.nodata is not None else nodata_default
        data = src.read(1, window=win, boundless=True, fill_value=nod)
        prof = src.profile.copy()
        prof.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                    transform=src.window_transform(win), compress="deflate",
                    tiled=True, blockxsize=512, blockysize=512, nodata=nod)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(data, 1)


def main(argv=None):
    params = load_params()
    cfg = params["sources"]["nlcd"]
    years = list(range(params["years"]["land_start"], params["years"]["land_end"] + 1))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", type=int, default=years)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--probe", action="store_true", help="only report which candidate URLs answer")
    args = ap.parse_args(argv)

    if args.probe:
        y = args.years[0]
        for cand, ok in probe(cfg, y):
            print(f"{'OK  ' if ok else 'miss'}  {cand.format(year=y)}")
        return

    todo = [y for y in args.years
            if args.overwrite or not path("interim", "nlcd", f"fctimp_{y}.tif").exists()]
    if not todo:
        log.info("all requested years already clipped")
        return
    template = resolve_template(cfg, todo[0])
    bounds = aoi_bounds(params["crs"]["nlcd"])
    failed = []
    for year in todo:
        out = path("interim", "nlcd", f"fctimp_{year}.tif")
        src = template.format(year=year)
        try:
            gdal_path, cleanup = open_source(src)
            clip_year(gdal_path, bounds, out, cfg["nodata"])
            log.info("wrote %s", out)
            if cleanup and cfg.get("delete_zip_after_clip", False):
                path("raw", "nlcd", cleanup.rsplit("/", 1)[-1], mkdir=False).unlink(missing_ok=True)
        except (rasterio.errors.RasterioIOError, requests.RequestException, StopIteration) as e:
            log.error("could not read %s (%s)", src, e)
            failed.append(year)
    if failed:
        log.warning("years not clipped: %s", failed)


if __name__ == "__main__":
    main()
