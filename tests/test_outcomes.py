import numpy as np
import pandas as pd

from src.panel import outcomes


def test_diversity_bounds():
    c = pd.DataFrame([[100, 0, 0], [50, 50, 0], [10, 10, 10], [np.nan, 1, 1], [0, 0, 0]])
    d = outcomes.diversity(c)
    assert d.loc[0, "entropy"] == 0 and d.loc[0, "simpson"] == 0
    assert np.isclose(d.loc[1, "simpson"], 0.5)
    assert np.isclose(d.loc[2, "entropy"], 1.0)
    assert d.loc[[3, 4]].isna().all().all()


def test_composition_five_group_ignores_missing_multi(params):
    race = pd.DataFrame(dict(ncessch=["a", "b"], year=[2005, 2015],
                             enr_white=[50, 40], enr_black=[50, 40], enr_hispanic=[0, 10],
                             enr_asian_pi=[0, 0], enr_aian=[0, 0], enr_multi=[np.nan, 10]))
    out = outcomes.composition(race, params)
    assert out.loc[0, "entropy5"] > 0 and np.isnan(out.loc[0, "entropy6"])
    assert not np.isnan(out.loc[1, "entropy6"])
    assert np.isclose(out.loc[1, "share_multi"], 0.1)
