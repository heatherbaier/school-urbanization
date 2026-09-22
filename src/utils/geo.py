"""Small geometry helpers shared by the panel and exposure steps."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd


def points_gdf(df: pd.DataFrame, lon="longitude", lat="latitude", crs="EPSG:4326") -> gpd.GeoDataFrame:
    """Rows with missing coordinates get an empty geometry, not dropped."""
    geom = gpd.points_from_xy(df[lon], df[lat], crs=crs)
    return gpd.GeoDataFrame(df.copy(), geometry=geom, crs=crs)


def projected_xy(df: pd.DataFrame, crs: str, lon="longitude", lat="latitude") -> pd.DataFrame:
    """Return df with x/y columns (metres) in the given projected CRS."""
    g = points_gdf(df, lon=lon, lat=lat).to_crs(crs)
    out = df.copy()
    out["x"] = g.geometry.x.to_numpy()
    out["y"] = g.geometry.y.to_numpy()
    return out


def dist(x1, y1, x2, y2) -> np.ndarray:
    return np.hypot(np.asarray(x1, float) - np.asarray(x2, float),
                    np.asarray(y1, float) - np.asarray(y2, float))


def snap_window(win):
    """Expand a fractional rasterio Window outward to whole cells.

    Written out by hand because Window.round_offsets/round_lengths changed
    signature across rasterio versions.
    """
    import math

    from rasterio.windows import Window

    c0, r0 = math.floor(win.col_off), math.floor(win.row_off)
    c1 = math.ceil(win.col_off + win.width)
    r1 = math.ceil(win.row_off + win.height)
    return Window(c0, r0, c1 - c0, r1 - r0)
