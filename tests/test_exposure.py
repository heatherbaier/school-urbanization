import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.exposure import buffers, smod


def _write(fp, arr, transform, crs, nodata=250):
    with rasterio.open(fp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="uint8", crs=crs, transform=transform, nodata=nodata) as d:
        d.write(arr.astype("uint8"), 1)


def test_buffer_means_on_synthetic_grid(tmp_path, params):
    # 30 m grid in the projected CRS itself, so the circle maps exactly.
    params = {**params, "crs": {**params["crs"]}}
    crs = params["crs"]["projected"]
    x0, y0 = 700_000, 3_740_000
    tr = from_origin(x0, y0 + 30 * 600, 30, 30)
    a = np.zeros((600, 600))
    a[:, 300:] = 100                      # east half fully impervious
    b = a.copy()
    b[:10, :10] = 250                     # nodata far from the site
    _write(tmp_path / "fctimp_2001.tif", a, tr, crs)
    _write(tmp_path / "fctimp_2002.tif", b, tr, crs)
    sites = pd.DataFrame(dict(site_id=["s_0"], x=[x0 + 300 * 30], y=[y0 + 300 * 30]))
    out = buffers.extract({2001: tmp_path / "fctimp_2001.tif", 2002: tmp_path / "fctimp_2002.tif"},
                          sites, params)
    for b_km in params["exposure"]["buffers_km"]:
        v = out[(out.year == 2001) & (out.buffer_km == b_km)].iloc[0]
        assert abs(v.imp_mean - 0.5) < 0.02
        assert v.valid_frac == 1.0
        # Circle area check: number of 30 m cells ~ pi r^2 / 900.
        assert abs(v.n_cells - np.pi * (b_km * 1000) ** 2 / 900) / v.n_cells < 0.01


def test_smod_distance_and_class():
    g = np.full((20, 20), 11)
    g[0, 0] = 30
    g[10, 10] = 21
    d = smod.distance_km(g, [21, 22, 23, 30], 1000)
    assert d[10, 10] == 0 and np.isclose(d[10, 13], 3)
    assert smod.degurba_l1([30, 22, 12, 10]).tolist() == [3, 2, 1, 0]


def test_nlcd_clip_from_local_zip(tmp_path, params):
    import zipfile

    from src.exposure import nlcd
    crs = params["crs"]["nlcd"]
    tr = from_origin(1_000_000, 1_300_000, 30, 30)
    a = np.arange(100 * 100).reshape(100, 100) % 101
    tif = tmp_path / "Annual_NLCD_FctImp_2001_CU_C1V2.tif"
    _write(tif, a, tr, crs)
    zp = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(tif, tif.name)
    gdal_path, cleanup = nlcd.open_source(str(zp))
    assert gdal_path.startswith("/vsizip/") and cleanup is None
    out = tmp_path / "clip.tif"
    # Bounds covering rows 10-19 and cols 20-29.
    nlcd.clip_year(gdal_path, (1_000_600, 1_300_000 - 600, 1_000_900, 1_300_000 - 300), out, 250)
    with rasterio.open(out) as d:
        np.testing.assert_array_equal(d.read(1), a[10:20, 20:30])
