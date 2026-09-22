"""Callaway–Sant'Anna event study on the land-based treatment (plan step 8).

Unconditional (no covariates) group-time ATTs following Callaway and
Sant'Anna (2021), written out directly because the panel is unbalanced by
design (openings and closures stay in) and the estimator is short.

For cohort g (schools first treated in year g) and year t,

    ATT(g, t) = mean over treated units in g of [Y_t - Y_{g-1}]
              - mean over comparison units of [Y_t - Y_{g-1}]

using a universal base period g-1, so pre-period estimates are cumulative
differences relative to the year before the event. Comparison units are
never-treated fringe schools, plus (control_group: notyet) fringe schools
not yet treated by max(t, g-1). A unit enters a cell only if it has Y in
both t and g-1.

Event-time estimates average ATT(g, g+e) across cohorts, weighted by the
number of treated units in each cell. The post-period summary averages the
event-time estimates for e >= 0. Inference is a cluster bootstrap that
resamples school districts, and the few-clusters caveat applies.

These are descriptive effects of envelopment on who attends a school.
Composition changes mostly reflect sorting, not school performance.

Writes
    outputs/tables/es_<outcome>.csv     event-time estimates
    outputs/tables/es_summary.csv       post-period average per outcome
    outputs/figures/es_outcomes.png     small multiples
    outputs/event_study.md              short report

    python -m src.analysis.event_study
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import load_params, path  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

log = get_logger(__name__)

INK, INK_2, MARK, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df", "#fcfcfb"

LABELS = {
    "log_enr": "log enrollment",
    "entropy5": "diversity (entropy, 5 groups)",
    "simpson5": "diversity (Simpson, 5 groups)",
    "share_white": "share white",
    "share_black": "share Black",
    "share_hispanic": "share Hispanic",
    "share_asian_pi": "share Asian / Pacific Islander",
}


def analysis_sample(panel: pd.DataFrame, treat: pd.DataFrame, outcomes) -> pd.DataFrame:
    """One row per school_uid x year for primary fringe schools, with G
    (event year, 0 for never treated) and a district cluster id."""
    t = treat[treat["in_primary"] & treat["group"].isin(["treated", "never_treated"])]
    t = t.assign(G=t["g_land"].fillna(0).astype(int))[["school_uid", "G"]]
    cols = ["school_uid", "year", "leaid", *[c for c in outcomes if c in panel]]
    p = (panel[cols].sort_values(["school_uid", "year"])
         .drop_duplicates(["school_uid", "year"])
         .merge(t, on="school_uid", how="inner"))
    # Cluster on the district the school belonged to in its first year.
    first_lea = p.groupby("school_uid")["leaid"].first()
    p["cluster"] = p["school_uid"].map(first_lea).fillna(p["school_uid"])
    return p


def _wide(sample: pd.DataFrame, y: str) -> tuple[pd.DataFrame, pd.Series]:
    w = sample.pivot_table(index="school_uid", columns="year", values=y, aggfunc="first")
    g = sample.groupby("school_uid")["G"].first().reindex(w.index)
    return w, g


def att_gt(wide: pd.DataFrame, G: pd.Series, window, control_group: str,
           min_cell: int = 1) -> pd.DataFrame:
    """Group-time ATTs on a school x year wide matrix."""
    years = set(wide.columns)
    Gv = G.to_numpy()
    rows = []
    for g in sorted(G[G > 0].unique()):
        base = g - 1
        if base not in years:
            continue
        yb = wide[base].to_numpy()
        is_g = Gv == g
        for e in window:
            t = g + e
            if t not in years or t == base:
                continue
            yt = wide[t].to_numpy()
            d = yt - yb
            ok = ~np.isnan(d)
            if control_group == "never":
                ctrl = Gv == 0
            else:
                ctrl = (Gv == 0) | (Gv > max(t, base))
            tr_d, c_d = d[is_g & ok], d[ctrl & ok & ~is_g]
            if len(tr_d) < min_cell or len(c_d) < min_cell:
                continue
            rows.append((g, t, e, tr_d.mean() - c_d.mean(), len(tr_d), len(c_d)))
    return pd.DataFrame(rows, columns=["g", "t", "e", "att", "n_treated", "n_control"])


def aggregate(gt: pd.DataFrame) -> pd.DataFrame:
    """Event-time estimates, weighting cohorts by treated units in the cell."""
    if gt.empty:
        return pd.DataFrame(columns=["e", "estimate", "n_treated", "n_cohorts"])
    agg = gt.groupby("e").apply(
        lambda d: pd.Series({"estimate": np.average(d["att"], weights=d["n_treated"]),
                             "n_treated": d["n_treated"].sum(), "n_cohorts": len(d)}),
        include_groups=False)
    agg = agg.reset_index()
    base = pd.DataFrame({"e": [-1], "estimate": [0.0], "n_treated": [np.nan], "n_cohorts": [np.nan]})
    return pd.concat([agg, base]).sort_values("e").reset_index(drop=True)


def post_average(es: pd.DataFrame) -> float:
    post = es[es["e"] >= 0]
    return float(post["estimate"].mean()) if len(post) else np.nan


def estimate(sample: pd.DataFrame, y: str, cfg: dict, seed: int = 0):
    window = range(cfg["window"][0], cfg["window"][1] + 1)
    wide, G = _wide(sample, y)
    gt = att_gt(wide, G, window, cfg["control_group"], cfg.get("min_cell", 1))
    es = aggregate(gt)
    avg = post_average(es)

    # Cluster bootstrap over districts.
    rng = np.random.default_rng(seed)
    clusters = sample.groupby("school_uid")["cluster"].first().reindex(wide.index)
    by_cluster = {c: np.nonzero(clusters.to_numpy() == c)[0] for c in clusters.unique()}
    keys = list(by_cluster)
    draws, avg_draws = [], []
    for _ in range(cfg["n_boot"]):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([by_cluster[keys[k]] for k in pick])
        b_wide = wide.iloc[idx].reset_index(drop=True)
        b_G = G.iloc[idx].reset_index(drop=True)
        b_es = aggregate(att_gt(b_wide, b_G, window, cfg["control_group"], cfg.get("min_cell", 1)))
        draws.append(b_es.set_index("e")["estimate"])
        avg_draws.append(post_average(b_es))
    B = pd.concat(draws, axis=1)
    a = (1 - cfg["ci_level"]) / 2
    es = es.set_index("e")
    es["se"] = B.std(axis=1)
    es["ci_lo"] = B.quantile(a, axis=1)
    es["ci_hi"] = B.quantile(1 - a, axis=1)
    es.loc[-1, ["se", "ci_lo", "ci_hi"]] = 0.0
    avg_draws = np.asarray(avg_draws, float)
    summary = dict(outcome=y, post_average=avg, se=np.nanstd(avg_draws, ddof=1),
                   ci_lo=np.nanquantile(avg_draws, a), ci_hi=np.nanquantile(avg_draws, 1 - a),
                   n_treated=int((G > 0).sum()), n_never=int((G == 0).sum()),
                   n_clusters=len(keys))
    pre = es[es.index < -1]
    summary["pre_mean"] = float(pre["estimate"].mean()) if len(pre) else np.nan
    return es.reset_index(), gt, summary


def plot(results: dict, fp, cfg):
    n = len(results)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, (y, es) in zip(axes.flat, results.items()):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.axvline(-0.5, color=GRIDC, linewidth=1, linestyle="--")
        ax.fill_between(es["e"], es["ci_lo"], es["ci_hi"], color=MARK, alpha=0.18, linewidth=0)
        ax.plot(es["e"], es["estimate"], color=MARK, linewidth=2, marker="o", markersize=4)
        ax.set_title(LABELS.get(y, y), loc="left", color=INK, fontsize=10)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.grid(axis="y", color=GRIDC, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRIDC)
    for ax in axes[-1]:
        ax.set_xlabel("years relative to event", color=INK_2, fontsize=9)
    fig.suptitle(f"Event study, land-based envelopment ({int(cfg['ci_level'] * 100)}% cluster-bootstrap CI)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(fp, dpi=150)
    plt.close(fig)


def _md(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep, *("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))])


def main():
    params = load_params()
    cfg = params["analysis"]["event_study"]
    panel = pd.read_parquet(path("processed", "school_year_panel.parquet"))
    treat = pd.read_parquet(path("processed", "treatment.parquet"))
    sample = analysis_sample(panel, treat, cfg["outcomes"])
    log.info("sample: %d schools (%d treated), %d clusters", sample["school_uid"].nunique(),
             sample.loc[sample["G"] > 0, "school_uid"].nunique(), sample["cluster"].nunique())

    results, summaries = {}, []
    for y in cfg["outcomes"]:
        if y not in sample:
            log.warning("outcome %s not in panel; skipping", y)
            continue
        es, gt, summ = estimate(sample, y, cfg)
        es.to_csv(path("tables", f"es_{y}.csv"), index=False)
        gt.to_csv(path("tables", f"es_{y}_gt.csv"), index=False)
        results[y] = es
        summaries.append(summ)
        log.info("%s: post average %.4f [%.4f, %.4f]", y, summ["post_average"], summ["ci_lo"], summ["ci_hi"])
    summary = pd.DataFrame(summaries)
    summary.to_csv(path("tables", "es_summary.csv"), index=False)
    fig_fp = path("figures", "es_outcomes.png")
    plot(results, fig_fp, cfg)

    report_dir = fig_fp.parent.parent
    d = params["definitions"]["land_event"]
    rule = (f"a rise of at least {d['change']} above baseline" if d.get("mode") == "change"
            else f"reaching {d['threshold']}")
    report = f"""# Event study — {params['pilot']}

Treatment is the land-based event ({d['buffer_km']} km buffer, {rule}, held {d['persistence_years']} years).
Comparison units are {'never-treated and not-yet-treated' if cfg['control_group'] == 'notyet' else 'never-treated'} fringe schools.
Estimates use a universal base period (the year before the event, fixed at zero).
Intervals come from a cluster bootstrap over school districts ({cfg['n_boot']} draws).

These estimates describe how the student body changes as land around a school develops.
They mostly reflect who moves in, not how the school performs.

![event study]({os.path.relpath(fig_fp, report_dir).replace(os.sep, '/')})

## Post-period averages

`pre_mean` is the average of the pre-period estimates (e ≤ −2) and should be near zero.

{_md(summary)}

Event-time tables are in `outputs/tables/es_<outcome>.csv`, and the group-time
cells behind them in `es_<outcome>_gt.csv`.
"""
    (report_dir / "event_study.md").write_text(report)
    log.info("wrote %s", report_dir / "event_study.md")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
