import numpy as np
import pandas as pd

from src.exposure import events


def test_land_event_persistence_and_censoring():
    yrs = np.arange(2000, 2011)
    # Blip in 2003 that does not persist, real crossing in 2006.
    v = np.array([.05, .06, .08, .30, .10, .12, .26, .28, .31, .33, .35])
    assert events.land_event(yrs, v, 2000, 0.25, 3) == (2006.0, False)
    # Crossing in the last two years cannot be confirmed with persistence 3.
    v2 = np.array([.05] * 9 + [.3, .3])
    g, cens = events.land_event(yrs, v2, 2000, 0.25, 3)
    assert np.isnan(g) and cens
    # Never crosses.
    g, cens = events.land_event(yrs, np.full(11, .1), 2000, 0.25, 3)
    assert np.isnan(g) and not cens
    # Crossings at or before baseline do not count.
    g, _ = events.land_event(yrs, np.full(11, .5), 2010, 0.25, 3)
    assert np.isnan(g)


def test_pop_event():
    ep = np.array([1990, 2000, 2010, 2020])
    assert events.pop_event(ep, np.array([11, 12, 21, 30]), 2003, [11, 12, 13], [21, 22, 23, 30]) == 2010
    assert np.isnan(events.pop_event(ep, np.array([21, 21, 30, 30]), 2003, [11, 12, 13], [21, 22, 23, 30]))


def test_build_groups(params):
    years = range(1995, 2021)
    panel = pd.DataFrame([dict(school_uid=u, site_id=f"{u}_0", year=y, in_primary=True, enr=100)
                          for u in ("treat", "never", "urban") for y in range(2000, 2021)])
    # "late" opens in 2008: first-year baseline, land series exists before opening.
    panel = pd.concat([panel, pd.DataFrame([dict(school_uid="late", site_id="late_0", year=y,
                                                  in_primary=True, enr=50) for y in range(2008, 2021)])])
    traj = {"treat": lambda y: .05 if y < 2010 else .4, "never": lambda y: .05,
            "urban": lambda y: .6, "late": lambda y: .05 if y < 2012 else .3}
    imp = pd.DataFrame([dict(site_id=f"{u}_0", year=y, buffer_km=b, imp_mean=f(y))
                        for u, f in traj.items() for y in years for b in (1, 2, 5)])
    smod = pd.DataFrame([dict(site_id=f"{u}_0", epoch=e, smod_code=12 if e < 2015 or u == "never" else 21,
                              km_to_urban_cluster=5.0, projected=False)
                         for u in traj for e in (1990, 1995, 2000, 2005, 2010, 2015, 2020)])
    t = events.build(panel, imp, smod, params).set_index("school_uid")
    assert t.loc["treat", "group"] == "treated" and t.loc["treat", "g_land"] == 2010
    assert t.loc["treat", "n_pre_years"] == 10
    assert t.loc["never", "group"] == "never_treated"
    assert t.loc["urban", "group"] == "not_fringe"
    assert t.loc["late", "baseline_year"] == 2008 and t.loc["late", "g_land"] == 2012
    assert t.loc["late", "n_pre_years"] == 4
    assert t.loc["treat", "g_pop"] == 2015 and np.isnan(t.loc["never", "g_pop"])
