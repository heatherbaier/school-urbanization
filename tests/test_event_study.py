import numpy as np
import pandas as pd

from src.analysis import event_study as es


def _panel(effect=0.2, seed=0, pretrend=0.0):
    """Staggered synthetic panel with a known post-event effect."""
    rng = np.random.default_rng(seed)
    rows, treat = [], []
    for i in range(120):
        u = f"s{i:03d}"
        g = [2006, 2010, 2014][i % 3] if i < 60 else 0
        unit_fe = rng.normal(0, 1)
        for y in range(2000, 2021):
            yfe = 0.03 * (y - 2000)
            eff = effect if g and y >= g else 0.0
            pre = pretrend * (y - g) if g and y < g else 0.0
            rows.append(dict(school_uid=u, year=y, leaid=f"d{i % 15}",
                             log_enr=unit_fe + yfe + eff + pre + rng.normal(0, 0.02)))
        treat.append(dict(school_uid=u, in_primary=True, group="treated" if g else "never_treated",
                          g_land=float(g) if g else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(treat)


CFG = dict(window=[-5, 10], control_group="notyet", min_cell=1, n_boot=49, ci_level=0.95)


def test_recovers_known_effect():
    panel, treat = _panel(effect=0.2)
    s = es.analysis_sample(panel, treat, ["log_enr"])
    est, gt, summ = es.estimate(s, "log_enr", CFG)
    est = est.set_index("e")
    assert abs(summ["post_average"] - 0.2) < 0.02
    assert (est.loc[[-5, -4, -3, -2], "estimate"].abs() < 0.02).all()
    assert est.loc[-1, "estimate"] == 0
    assert (est.loc[0:5, "ci_lo"] > 0.1).all()
    assert summ["n_treated"] == 60 and summ["n_clusters"] == 15


def test_never_control_matches_and_pretrend_shows():
    panel, treat = _panel(effect=0.0, pretrend=0.05)
    s = es.analysis_sample(panel, treat, ["log_enr"])
    est, _, summ = es.estimate(s, "log_enr", {**CFG, "control_group": "never"})
    est = est.set_index("e")
    # A linear pre-trend relative to base e=-1 shows up as e * 0.05 before the event.
    assert abs(est.loc[-5, "estimate"] - (-0.2)) < 0.03


def test_excludes_censored_and_not_fringe():
    panel, treat = _panel()
    treat.loc[0, "group"] = "censored"
    treat.loc[1, "group"] = "not_fringe"
    s = es.analysis_sample(panel, treat, ["log_enr"])
    assert not s["school_uid"].isin(["s000", "s001"]).any()


def test_main_writes_outputs(tmp_path, params, monkeypatch):
    import copy

    import src.utils.config as config
    p = copy.deepcopy(params)
    for k in p["paths"]:
        p["paths"][k] = str(tmp_path / k)
    p["analysis"]["event_study"].update(n_boot=19, outcomes=["log_enr", "entropy5"])
    for mod in (config, es):
        monkeypatch.setattr(mod, "load_params", lambda *a, **k: p)
    panel, treat = _panel(effect=0.2)
    panel["entropy5"] = 0.5 + 0.1 * (panel["log_enr"] - panel["log_enr"].mean())
    panel.to_parquet(config.path("processed", "school_year_panel.parquet"))
    treat.to_parquet(config.path("processed", "treatment.parquet"))
    es.main()
    assert (tmp_path / "event_study.md").exists()
    assert (tmp_path / "figures" / "es_outcomes.png").exists()
    summ = pd.read_csv(config.path("tables", "es_summary.csv"))
    assert set(summ["outcome"]) == {"log_enr", "entropy5"}
