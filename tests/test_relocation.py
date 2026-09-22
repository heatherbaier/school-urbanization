import numpy as np
import pandas as pd

from src.panel import relocation


def _cfg(params):
    return params["relocation"]


def test_real_move_starts_new_site(params):
    xy = np.array([[0, 0], [10, 0], [3000, 0], [3010, 0], [3005, 0]], float)
    site, outl, move = relocation.assign_sites(xy, _cfg(params))
    assert site.tolist() == [0, 0, 1, 1, 1]
    assert not outl.any()
    assert np.isclose(move[2], 3000)


def test_geocode_blip_is_outlier_not_move(params):
    xy = np.array([[0, 0], [0, 0], [4000, 0], [20, 0], [0, 0]], float)
    site, outl, move = relocation.assign_sites(xy, _cfg(params))
    assert site.tolist() == [0, 0, 0, 0, 0]
    assert outl.tolist() == [False, False, True, False, False]
    assert np.isnan(move).all()


def test_missing_coordinates_are_imputed_from_site(params):
    df = pd.DataFrame(dict(
        ncessch=["130000100001"] * 4, year=[2000, 2001, 2002, 2003],
        x=[0, np.nan, 5, 3000], y=[0, np.nan, 0, 0]))
    uid = pd.DataFrame(dict(ncessch=["130000100001"], school_uid=["130000100001"]))
    ss, sites = relocation.build(df, uid, params)
    assert ss["site_id"].tolist() == ["130000100001_0"] * 3 + ["130000100001_1"]
    assert ss["coord_imputed"].tolist() == [False, True, False, False]
    assert len(sites) == 2
    assert sites.set_index("site_id").loc["130000100001_0", "x"] == 2.5
