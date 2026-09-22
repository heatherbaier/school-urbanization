"""Mean fractional impervious surface in radial buffers around each site.

For every site (a school at one location, see src/panel/relocation.py) and
every land-cover year, computes the mean impervious fraction (0-1) of the
30 m cells whose centres fall inside 1, 2 and 5 km buffers. Buffers are
drawn as circles in the projected CRS (UTM) and transformed to the raster
CRS, so points are reprojected instead of rasters being resampled.

Cell membership is computed once per site and buffer and reused for every
year, which is valid because all clipped years share one grid.

Output: data/interim/exposure/impervious_buffers.parquet with
    site_id, year, buffer_km, imp_mean, valid_frac, n_cells
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

from src.utils.config import load_params, path
from src.utils.geo import snap_window
from src.utils.log import get_logger

log = get_logger(__name__)


def buffer_cells(sites: pd.DataFrame, buffers_km, proj_crs: str, transform, shape, raster_crs):
    """Map (site_id, buffer_km) -> (flat indices of in-raster cells inside the
    buffer, number of cells the full buffer covers).

    Sites with no usable coordinate, or whose buffer misses the raster, get
    an empty index array, so their values come out NaN instead of failing.
    """
    ok = np.isfinite(sites["x"].to_numpy(float)) & np.isfinite(sites["y"].to_numpy(float))
    if (~ok).any():
        log.warning("%d sites have no usable coordinate: %s", int((~ok).sum()),
                    ", ".join(sites.loc[~ok, "site_id"].astype(str).head(10)))
    pts = gpd.GeoSeries(gpd.points_from_xy(sites["x"], sites["y"]), crs=proj_crs)
    empty = np.array([], dtype=np.int64)
    cells = {}
    partial, missing = set(), set()
    for b in buffers_km:
        polys = pts.buffer(b * 1000, resolution=64).to_crs(raster_crs)
        for sid, poly, good in zip(sites["site_id"], polys, ok):
            bounds = poly.bounds if good and not poly.is_empty else None
            if bounds is None or not np.all(np.isfinite(bounds)):
                cells[(sid, b)] = (empty, 0)
                continue
            win = snap_window(from_bounds(*bounds, transform=transform))
            r0, c0 = int(win.row_off), int(win.col_off)
            h, w = int(win.height), int(win.width)
            m = geometry_mask([poly], out_shape=(h, w), invert=True,
                              transform=rasterio.windows.transform(win, transform))
            rr, cc = np.nonzero(m)
            rr, cc = rr + r0, cc + c0
            inside = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            if not inside.any():
                missing.add(sid)
            elif not inside.all():
                partial.add(sid)
            cells[(sid, b)] = (np.ravel_multi_index((rr[inside], cc[inside]), shape), len(rr))
    if partial:
        log.warning("%d sites have a buffer running past the raster edge (e.g. %s); "
                    "they are kept only where valid_frac >= min_valid_frac",
                    len(partial), ", ".join(sorted(partial)[:5]))
    if missing:
        log.warning("%d sites lie entirely outside the raster (e.g. %s)",
                    len(missing), ", ".join(sorted(missing)[:5]))
    return cells


def extract(raster_paths: dict[int, Path], sites: pd.DataFrame, params: dict) -> pd.DataFrame:
    buffers_km = params["exposure"]["buffers_km"]
    valid_max = params["sources"]["nlcd"]["valid_max"]
    ref = None
    cells = None
    rows = []
    for year, fp in sorted(raster_paths.items()):
        with rasterio.open(fp) as src:
            grid = (src.transform, src.shape, src.crs)
            if ref is None:
                ref = grid
                cells = buffer_cells(sites, buffers_km, params["crs"]["projected"],
                                     src.transform, src.shape, src.crs)
            elif grid != ref:
                raise ValueError(f"{fp} is on a different grid from the first year")
            a = src.read(1).astype(np.float32).ravel()
            nod = src.nodata
        invalid = a > valid_max
        if nod is not None:
            invalid |= a == nod
        a[invalid] = np.nan
        for (sid, b), (idx, n_full) in cells.items():
            v = a[idx]
            ok = np.count_nonzero(~np.isnan(v))
            # valid_frac is relative to the whole buffer, so a buffer cut off
            # by the raster edge counts the missing part as invalid.
            rows.append((sid, year, b, np.nanmean(v) / 100 if ok else np.nan,
                         ok / n_full if n_full else 0.0, n_full))
        log.info("extracted %d", year)
    out = pd.DataFrame(rows, columns=["site_id", "year", "buffer_km", "imp_mean", "valid_frac", "n_cells"])
    out.loc[out["valid_frac"] < params["exposure"]["min_valid_frac"], "imp_mean"] = np.nan
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", nargs="+", type=int, default=None)
    args = ap.parse_args(argv)
    params = load_params()
    d = path("interim", "nlcd", mkdir=False)
    files = {int(p.stem.split("_")[-1]): p for p in sorted(d.glob("fctimp_*.tif"))}
    if args.years:
        files = {y: p for y, p in files.items() if y in set(args.years)}
    if not files:
        raise FileNotFoundError(f"no clipped rasters in {d}; run src.exposure.nlcd first")
    sites = pd.read_parquet(path("interim", "ccd", "sites.parquet"))
    out = extract(files, sites, params)
    fp = path("interim", "exposure", "impervious_buffers.parquet")
    out.to_parquet(fp, index=False)
    log.info("wrote %s (%d rows)", fp, len(out))


if __name__ == "__main__":
    main()
