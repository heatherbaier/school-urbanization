"""Pilot viability diagnostics (plan step 5).

Answers whether the design has enough to work with in Atlanta. It covers
how many fringe schools, how many treated, when events happen, how many
treated schools have clean pre-periods, how sensitive those counts are to
the definitions, and how well the outcomes cover the event window.

Writes
    data/processed/treatment.parquet           per-school definitions (defaults)
    outputs/tables/diag_*.csv                  every table in the report
    outputs/figures/diag_*.png                 three figures
    outputs/diagnostics.md                     the report

    python -m src.analysis.diagnostics
"""
from __future__ import annotations

import copy
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.exposure import events  # noqa: E402
from src.utils.config import load_params, path  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

log = get_logger(__name__)

# Sensitivity grid. Kept small so it runs in a minute or two.
GRID = {
    "buffer_km": [1, 2, 5],
    "threshold": [0.15, 0.20, 0.25, 0.30],
    "max_impervious": [0.05, 0.10, 0.15],
}
EVENT_WINDOW = range(-5, 11)
# MEPS columns are added from whatever the panel carries (names come from the API).
OUTCOMES = ["enr", "entropy5", "share_white", "prof_z_read", "prof_z_math"]

# Single-series figures use one hue on a light surface.
INK, INK_2, MARK, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df", "#fcfcfb"


def funnel(t: pd.DataFrame, primary_only: bool = True) -> pd.DataFrame:
    d = t[t["in_primary"]] if primary_only else t
    fr = d[d["is_fringe"]]
    tr = fr[fr["group"] == "treated"]
    rows = [
        ("schools", len(d)),
        ("with baseline impervious", int(d["imp_base"].notna().sum())),
        ("fringe at baseline", len(fr)),
        ("treated (land event)", len(tr)),
        ("treated with >= 3 pre years", int((tr["n_pre_years"] >= 3).sum())),
        ("treated with >= 5 pre years", int((tr["n_pre_years"] >= 5).sum())),
        ("never treated", int((fr["group"] == "never_treated").sum())),
        ("censored (crossed too late to confirm)", int((fr["group"] == "censored").sum())),
        ("fringe with population event", int(fr["g_pop"].notna().sum())),
    ]
    return pd.DataFrame(rows, columns=["step", "n"])


def event_years(t: pd.DataFrame) -> pd.DataFrame:
    tr = t[t["in_primary"] & (t["group"] == "treated")]
    return (tr.groupby("g_land").agg(n=("school_uid", "size"),
                                     n_pre3=("n_pre_years", lambda v: int((v >= 3).sum())))
            .reset_index().rename(columns={"g_land": "event_year"}))


def sensitivity(panel, imp, smod, params) -> pd.DataFrame:
    rows = []
    for b, thr, mx in itertools.product(GRID["buffer_km"], GRID["threshold"], GRID["max_impervious"]):
        if mx >= thr:
            continue
        defs = copy.deepcopy(params["definitions"])
        defs["fringe"]["buffer_km"] = b
        defs["fringe"]["max_impervious"] = mx
        defs["land_event"]["buffer_km"] = b
        defs["land_event"]["threshold"] = thr
        t = events.build(panel, imp, smod, params, defs)
        t = t[t["in_primary"] & t["is_fringe"]]
        tr = t[t["group"] == "treated"]
        rows.append(dict(buffer_km=b, threshold=thr, fringe_max_imp=mx, n_fringe=len(t),
                         n_treated=len(tr), n_treated_pre3=int((tr["n_pre_years"] >= 3).sum()),
                         n_treated_pre5=int((tr["n_pre_years"] >= 5).sum()),
                         n_never=int((t["group"] == "never_treated").sum()),
                         n_censored=int((t["group"] == "censored").sum())))
    return pd.DataFrame(rows)


def outcome_coverage(panel: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Share of treated schools observed with each outcome at each event time."""
    tr = t.loc[t["in_primary"] & (t["group"] == "treated"), ["school_uid", "g_land"]]
    p = panel.merge(tr, on="school_uid")
    p["rel_year"] = p["year"] - p["g_land"]
    p = p[p["rel_year"].isin(EVENT_WINDOW)]
    cols = [c for c in OUTCOMES if c in p] + [c for c in p.columns if c.startswith("meps")][:1]
    n = len(tr)
    cov = p.groupby("rel_year")[cols].agg(lambda v: v.notna().sum() / n if n else np.nan)
    cov.insert(0, "n_schools_observed", p.groupby("rel_year")["school_uid"].nunique() / max(n, 1))
    return cov.reindex(EVENT_WINDOW).reset_index()


def land_vs_pop(t: pd.DataFrame) -> pd.DataFrame:
    fr = t[t["in_primary"] & t["is_fringe"]]
    return pd.crosstab(fr["g_land"].notna().map({True: "land event", False: "no land event"}),
                       fr["g_pop"].notna().map({True: "pop event", False: "no pop event"}))


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(axis="y", color=GRIDC, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ylabel == "schools":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def figures(t, imp, panel, params):
    defs = params["definitions"]
    out = {}

    ev = event_years(t)
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    if len(ev):
        ax.bar(ev["event_year"], ev["n"], width=0.8, color=MARK, edgecolor=SURFACE, linewidth=1)
    _style(ax, "Treated fringe schools by land-event year", "event year", "schools")
    fig.tight_layout()
    out["event_years"] = path("figures", "diag_event_years.png")
    fig.savefig(out["event_years"], dpi=150)
    plt.close(fig)

    d = t[t["in_primary"] & t["imp_base"].notna()]
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    ax.hist(d["imp_base"], bins=np.linspace(0, 1, 41), color=MARK, edgecolor=SURFACE, linewidth=1)
    ax.axvline(defs["fringe"]["max_impervious"], color=INK, linewidth=1.2, linestyle="--")
    ax.text(defs["fringe"]["max_impervious"] + 0.01, ax.get_ylim()[1] * 0.92,
            f"fringe cutoff {defs['fringe']['max_impervious']:.2f}", color=INK, fontsize=9)
    _style(ax, f"Baseline impervious fraction, {defs['fringe']['buffer_km']} km buffer",
           "impervious fraction", "schools")
    ax.xaxis.set_major_locator(plt.AutoLocator())
    fig.tight_layout()
    out["baseline_imp"] = path("figures", "diag_baseline_impervious.png")
    fig.savefig(out["baseline_imp"], dpi=150)
    plt.close(fig)

    tr = t.loc[t["in_primary"] & (t["group"] == "treated"), ["school_uid", "g_land"]]
    le = defs["land_event"]
    sites = events.site_by_year(panel[panel["school_uid"].isin(tr["school_uid"])],
                                sorted(imp["year"].unique()))
    ser = events.exposure_series(sites, imp, le["buffer_km"]).merge(tr, on="school_uid")
    ser["rel_year"] = ser["year"] - ser["g_land"]
    ser = ser[ser["rel_year"].between(-10, 10)]
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    if len(ser):
        q = ser.groupby("rel_year")["imp_mean"].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=MARK, alpha=0.18, linewidth=0)
        ax.plot(q.index, q[0.5], color=MARK, linewidth=2)
    ax.axhline(le["threshold"], color=INK, linewidth=1.2, linestyle="--")
    ax.axvline(0, color=GRIDC, linewidth=1)
    _style(ax, f"Impervious fraction around the event, treated schools ({le['buffer_km']} km)",
           "years relative to event", "median and interquartile range")
    fig.tight_layout()
    out["trajectory"] = path("figures", "diag_event_trajectory.png")
    fig.savefig(out["trajectory"], dpi=150)
    plt.close(fig)
    return out


def _md(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main():
    params = load_params()
    panel, imp, smod = events.load_inputs()
    t = events.build(panel, imp, smod, params)
    t.to_parquet(path("processed", "treatment.parquet"), index=False)

    tables = {
        "funnel": funnel(t),
        "funnel_all_schools": funnel(t, primary_only=False),
        "event_years": event_years(t),
        "sensitivity": sensitivity(panel, imp, smod, params),
        "outcome_coverage": outcome_coverage(panel, t),
    }
    lvp = land_vs_pop(t)
    for k, v in tables.items():
        v.to_csv(path("tables", f"diag_{k}.csv"), index=False)
    lvp.to_csv(path("tables", "diag_land_vs_pop.csv"))
    figs = figures(t, imp, panel, params)

    d = params["definitions"]
    # The report sits next to the figures folder (outputs/ by default), and
    # links to the figures relative to itself.
    report_dir = path("figures", mkdir=True).parent
    rel = lambda p: os.path.relpath(p, report_dir).replace(os.sep, "/")  # noqa: E731
    report = f"""# Pilot diagnostics — {params['pilot']}

Primary sample only (regular, non-charter, non-virtual schools) unless stated.

## Definitions used

- Baseline is the school's {'first panel year' if d['baseline']['mode'] == 'first_year' else 'year ' + str(d['baseline']['fixed_year'])}
- Fringe means impervious below {d['fringe']['max_impervious']} in the {d['fringe']['buffer_km']} km buffer and within {d['fringe']['max_km_to_urban_cluster']} km of a GHS-SMOD urban cluster at baseline
- The land event is the first year the {d['land_event']['buffer_km']} km buffer reaches {d['land_event']['threshold']} and stays there for {d['land_event']['persistence_years']} years

## Sample funnel

{_md(tables['funnel'])}

## Event years

![event years]({rel(figs['event_years'])})

{_md(tables['event_years'].astype({'event_year': int}))}

## Baseline impervious distribution

![baseline impervious]({rel(figs['baseline_imp'])})

## Impervious trajectory around the event

This is a construction check. The median should cross the dashed threshold at 0 and stay above it.

![trajectory]({rel(figs['trajectory'])})

## Sensitivity of counts to the definitions

{_md(tables['sensitivity'])}

## Outcome coverage in the event window

Share of treated schools with a non-missing value at each year relative to the event.

{_md(tables['outcome_coverage'])}

## Land versus population events among fringe schools

{_md(lvp.reset_index())}
"""
    fp = report_dir / "diagnostics.md"
    fp.write_text(report)
    log.info("wrote %s", fp)
    print(tables["funnel"].to_string(index=False))


if __name__ == "__main__":
    main()
