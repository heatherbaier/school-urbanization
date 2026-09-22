"""Run directory -> crosswalk -> relocation -> panel on a tiny fake Urban pull."""
import copy

import numpy as np
import pandas as pd
import pytest

import src.utils.config as config
from src.panel import build_panel, crosswalk, directory, relocation


@pytest.fixture
def sandbox(tmp_path, params, monkeypatch):
    p = copy.deepcopy(params)
    for k in p["paths"]:
        p["paths"][k] = str(tmp_path / k)
    p["years"]["school_start"], p["years"]["school_end"] = 2000, 2003
    p["sources"]["urban_api"]["enrollment_grades"] = [1, 99]
    p["sources"]["urban_api"]["edfacts_grades"] = [3]
    for mod in (config, directory, crosswalk, relocation, build_panel):
        monkeypatch.setattr(mod, "load_params", lambda *a, **k: p)
    return p


def _write(name, year, df):
    df.to_parquet(config.path("raw", "urban", name, f"{year}.parquet"), index=False)


def test_end_to_end(sandbox):
    for y in range(2000, 2004):
        d = pd.DataFrame(dict(
            ncessch=[130000100001, 130000200002, 139999900009],
            year=y, school_name=["Oak Elementary", "Pine Middle", "Far Away High"],
            leaid=["1300001", "1300002", "1399999"], lea_name="x", state_leaid="x",
            seasch=["1", "2", "9"], county_code=[13121, 13135, 13001],
            latitude=[33.75, 33.95, 31.0], longitude=[-84.39, -84.0, -82.0],
            school_type=1, school_status=1, charter=[0, 0, 0], virtual=0,
            enrollment=[500, 800, 300], teachers_fte=[25, 40, 20],
            free_or_reduced_price_lunch=[200, -3, 100]))
        if y == 2003:  # Pine Middle closes after 2002
            d = d[d.ncessch != 130000200002]
        _write("ccd_directory", y, d)
        ids = d.ncessch.tolist()
        _write("ccd_enrollment_grade/grade-1", y,
               pd.DataFrame(dict(ncessch=ids, year=y, grade=1, race=99, sex=99, enrollment=80)))
        _write("ccd_enrollment_grade/grade-99", y,
               pd.DataFrame(dict(ncessch=ids, year=y, grade=99, race=99, sex=99, enrollment=d.enrollment)))
        race = pd.DataFrame([dict(ncessch=i, year=y, grade=99, race=r, sex=99, enrollment=e)
                             for i in ids for r, e in [(1, 100), (2, 200), (3, 50), (4, 5), (5, 0), (7, -2 if y < 2002 else 10), (99, 360)]])
        _write("ccd_enrollment_race", y, race)
        _write("edfacts_assessments/grade-3", y, pd.DataFrame(dict(
            ncessch=ids, year=y, grade_edfacts=3, race=99, sex=99,
            read_test_num_valid=50, read_test_pct_prof_midpt=np.linspace(40, 80, len(ids)),
            math_test_num_valid=50, math_test_pct_prof_midpt=60)))

    directory.main()
    crosswalk.main()
    relocation.main()
    build_panel.main()
    panel = pd.read_parquet(config.path("processed", "school_year_panel.parquet"))

    # Out-of-area school dropped (no counties.gpkg, so selection is by county code).
    assert set(panel.ncessch) == {"130000100001", "130000200002"}
    assert len(panel) == 4 + 3
    oak = panel[panel.ncessch == "130000100001"].set_index("year")
    assert oak.enr_g01.eq(80).all() and oak.enr_total.eq(500).all()
    assert np.isnan(oak.loc[2000, "entropy6"]) and not np.isnan(oak.loc[2003, "entropy6"])
    assert oak.entropy5.notna().all()
    assert np.isclose(oak.loc[2001, "enr_growth"], 0)
    pine = panel[panel.ncessch == "130000200002"]
    assert pine.closed_in_window.all() and pine.is_closing_year.sum() == 1
    assert pine.frl_share.isna().all()          # -3 suppressed code became NaN
    assert panel.site_id.notna().all()
    assert "prof_z_read" in panel and panel.prof_z_math.dropna().eq(panel.prof_z_math.dropna()).all()


def test_coordinates_outside_aoi_are_blanked(sandbox):
    import geopandas as gpd
    from shapely.geometry import box

    aoi = gpd.GeoDataFrame(geometry=[box(-85, 33, -83.5, 34.5)], crs="EPSG:4326")
    aoi.to_file(config.path("interim", "study_area", "aoi.gpkg"), driver="GPKG")
    df = pd.DataFrame(dict(ncessch=["a", "a", "b"], year=[2000, 2001, 2000],
                           latitude=[33.75, 40.0, 95.0], longitude=[-84.39, -84.39, -84.0], leaid="1300001"))
    df = directory.clean(df.assign(county_code=13121))
    out = directory.null_outside_aoi(df)
    assert out["coord_outside_aoi"].tolist() == [False, True, False]
    assert out["latitude"].isna().tolist() == [False, True, True]
