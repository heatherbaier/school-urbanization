"""Run the diagnostics report end to end on a small synthetic panel."""
import copy

import numpy as np
import pandas as pd

import src.utils.config as config
from src.analysis import diagnostics
from src.exposure import events


def test_diagnostics_runs(tmp_path, params, monkeypatch):
    p = copy.deepcopy(params)
    for k in p["paths"]:
        p["paths"][k] = str(tmp_path / k)
    for mod in (config, diagnostics, events):
        monkeypatch.setattr(mod, "load_params", lambda *a, **k: p)

    rng = np.random.default_rng(0)
    uids = [f"s{i}" for i in range(40)]
    panel = pd.DataFrame([dict(school_uid=u, site_id=f"{u}_0", ncessch=u, year=y, in_primary=True,
                               enr=100 + rng.integers(0, 50), entropy5=0.5, share_white=0.5,
                               meps_poverty=0.2)
                          for u in uids for y in range(2000, 2023)])
    onset = {u: (2004 + i % 12 if i < 25 else 9999) for i, u in enumerate(uids)}
    imp = pd.DataFrame([dict(site_id=f"{u}_0", year=y, buffer_km=b,
                             imp_mean=0.05 + (0.35 if y >= onset[u] else 0) + rng.normal(0, .01))
                        for u in uids for y in range(1995, 2024) for b in (1, 2, 5)])
    smod = pd.DataFrame([dict(site_id=f"{u}_0", epoch=e, smod_code=12, km_to_urban_cluster=3.0,
                              projected=e > 2020)
                         for u in uids for e in range(1990, 2031, 5)])
    panel.to_parquet(config.path("processed", "school_year_panel.parquet"))
    imp.to_parquet(config.path("interim", "exposure", "impervious_buffers.parquet"))
    smod.to_parquet(config.path("interim", "exposure", "smod_sites.parquet"))

    # Two cached pre-2000 directory years; one school's 1998 geocode is 2 km off.
    early = pd.DataFrame([dict(ncessch=130000100000 + i, year=y, school_name=f"s{i}", leaid="1300001",
                               county_code=13121, latitude=33.75 + i * 0.01 + (0.018 if (i == 0 and y == 1998) else 0),
                               longitude=-84.39, enrollment=100)
                          for i in range(3) for y in (1998, 1999, 2000)])
    for y, g in early.groupby("year"):
        g.to_parquet(config.path("raw", "urban", "ccd_directory", f"{y}.parquet"), index=False)

    diagnostics.main()

    report = (tmp_path / "diagnostics.md").read_text()
    assert "Sample funnel" in report
    f = pd.read_csv(config.path("tables", "diag_funnel.csv")).set_index("step")["n"]
    assert f["fringe at baseline"] == 40
    assert f["treated (land event)"] == 25
    assert f["never treated"] == 15
    cov = pd.read_csv(config.path("tables", "diag_outcome_coverage.csv"))
    assert "meps_poverty" in cov.columns
    assert (tmp_path / "figures" / "diag_event_years.png").exists()
    sens = pd.read_csv(config.path("tables", "diag_sensitivity.csv"))
    assert set(sens["mode"]) == {"change", "level"}
    ec = pd.read_csv(config.path("tables", "diag_early_ccd_coverage.csv")).set_index("year")
    assert list(ec.index) == [1998, 1999]
    assert ec.loc[1998, "share_over_500m_from_ref"] > 0 and ec.loc[1999, "median_m_from_ref"] < 1
    assert "CCD coverage before 2000" in report
