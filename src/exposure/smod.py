"""GHS-SMOD Degree of Urbanisation class for each site at each epoch.

Downloads each epoch's global 1 km GHS-SMOD raster, clips it to the AOI
(plus a margin so distances to clusters outside the counties are right),
and records per site
    smod_code            settlement class at the site's cell
    degurba_l1           3 city (urban centre), 2 town/semi-dense, 1 rural
    km_to_urban_cluster  distance to the nearest urban cluster or centre cell
    km_to_urban_centre   distance to the nearest urban centre cell
    projected            True for epochs after the last observed epoch

Output: data/interim/exposure/smod_sites.parquet
"""
from __future__ import annotations

import urllib.request
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt

from src.utils.config import load_params, path
from src.utils.geo import snap_window
from src.utils.log import get_logger

log = get_logger(__name__)

# Extra margin beyond the AOI, so the nearest cluster can lie outside it.
DIST_MARGIN_M = 30_000


def download_epoch(epoch: int, params) -> str:
    url = params["sources"]["ghs_smod"]["url_template"].format(epoch=epoch)
    zp = path("raw", "ghsl", url.rsplit("/", 1)[-1])
    if not zp.exists():
        log.info("downloading %s", url)
        urllib.request.urlretrieve(url, zp)
    with zipfile.ZipFile(zp) as z:
        tif = next(n for n in z.namelist() if n.lower().endswith(".tif"))
    return f"/vsizip/{zp}/{tif}"


def clip(src_path: str, params, epoch: int):
    aoi = gpd.read_file(path("interim", "study_area", "aoi.gpkg", mkdir=False))
    bounds = aoi.to_crs(params["crs"]["ghsl"]).buffer(DIST_MARGIN_M).total_bounds
    out = path("interim", "ghsl", f"smod_{epoch}.tif")
    with rasterio.open(src_path) as src:
        win = snap_window(from_bounds(*bounds, transform=src.transform))
        data = src.read(1, window=win, boundless=True, fill_value=src.nodata or 0)
        prof = src.profile.copy()
        prof.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                    transform=src.window_transform(win), compress="deflate")
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(data, 1)
    return out


def degurba_l1(code):
    code = np.asarray(code)
    return np.select([code == 30, (code >= 21) & (code <= 23), (code >= 11) & (code <= 13)],
                     [3, 2, 1], default=0)


def distance_km(grid: np.ndarray, targets, cell_m: float) -> np.ndarray:
    """Distance from each cell centre to the nearest target cell, in km."""
    is_target = np.isin(grid, targets)
    if not is_target.any():
        return np.full(grid.shape, np.inf)
    return distance_transform_edt(~is_target) * cell_m / 1000


def sample_sites(tif, sites: pd.DataFrame, params) -> pd.DataFrame:
    cfg = params["exposure"]["smod"]
    pts = gpd.GeoSeries(gpd.points_from_xy(sites["x"], sites["y"]),
                        crs=params["crs"]["projected"]).to_crs(params["crs"]["ghsl"])
    with rasterio.open(tif) as src:
        grid = src.read(1)
        cell_m = abs(src.transform.a)
        rows, cols = rasterio.transform.rowcol(src.transform, pts.x.to_numpy(), pts.y.to_numpy())
    rows, cols = np.asarray(rows), np.asarray(cols)
    d_cluster = distance_km(grid, cfg["urban_cluster_codes"], cell_m)
    d_centre = distance_km(grid, cfg["urban_centre_codes"], cell_m)
    code = grid[rows, cols]
    return pd.DataFrame({
        "site_id": sites["site_id"].to_numpy(),
        "smod_code": code,
        "degurba_l1": degurba_l1(code),
        "km_to_urban_cluster": d_cluster[rows, cols],
        "km_to_urban_centre": d_centre[rows, cols],
    })


def main():
    params = load_params()
    cfg = params["sources"]["ghs_smod"]
    sites = pd.read_parquet(path("interim", "ccd", "sites.parquet"))
    out = []
    for epoch in cfg["epochs"]:
        tif = path("interim", "ghsl", f"smod_{epoch}.tif")
        if not tif.exists():
            tif = clip(download_epoch(epoch, params), params, epoch)
        s = sample_sites(tif, sites, params)
        s["epoch"] = epoch
        s["projected"] = epoch > cfg["last_observed_epoch"]
        out.append(s)
        log.info("sampled epoch %d", epoch)
    res = pd.concat(out, ignore_index=True)
    fp = path("interim", "exposure", "smod_sites.parquet")
    res.to_parquet(fp, index=False)
    log.info("wrote %s", fp)


if __name__ == "__main__":
    main()
