"""Pull CCD directory, enrollment, MEPS poverty and EDFacts assessments.

Pulls are statewide (fips filter only) and cached raw. Restriction to the
study counties happens in src/panel, so the county list can change without
re-downloading.

    python -m src.ingest.ccd                 # everything in params.yaml
    python -m src.ingest.ccd --only directory enrollment
"""
from __future__ import annotations

import argparse

from src.ingest.urban_api import UrbanClient
from src.utils.config import load_params
from src.utils.log import get_logger

log = get_logger(__name__)


def pull_directory(client, params):
    y0, y1 = params["years"]["school_start"], params["years"]["school_end"]
    q = {"fips": int(params["study_area"]["state_fips"])}
    for year in range(y0, y1 + 1):
        client.fetch_cached("ccd_directory", f"schools/ccd/directory/{year}", year, q)


def pull_enrollment(client, params):
    """Grade totals (all races) and the all-grade total by race."""
    y0, y1 = params["years"]["school_start"], params["years"]["school_end"]
    q = {"fips": int(params["study_area"]["state_fips"])}
    grades = params["sources"]["urban_api"]["enrollment_grades"]
    for year in range(y0, y1 + 1):
        for g in grades:
            client.fetch_cached(f"ccd_enrollment_grade/grade-{g}",
                                f"schools/ccd/enrollment/{year}/grade-{g}", year, q)
        client.fetch_cached("ccd_enrollment_race",
                            f"schools/ccd/enrollment/{year}/grade-99/race", year, q)


def pull_meps(client, params):
    cfg = params["sources"]["urban_api"]
    y0 = max(cfg["meps_start"], params["years"]["school_start"])
    q = {"fips": int(params["study_area"]["state_fips"])}
    for year in range(y0, params["years"]["school_end"] + 1):
        client.fetch_cached("meps", f"schools/meps/{year}", year, q)


def pull_edfacts(client, params):
    """School-by-year proficiency rates. State tests change over time, so
    these are only usable after standardizing within state-year-grade-subject."""
    cfg = params["sources"]["urban_api"]
    y0 = max(cfg["edfacts_start"], params["years"]["school_start"])
    q = {"fips": int(params["study_area"]["state_fips"])}
    for year in range(y0, params["years"]["school_end"] + 1):
        for g in cfg["edfacts_grades"]:
            client.fetch_cached(f"edfacts_assessments/grade-{g}",
                                f"schools/edfacts/assessments/{year}/grade-{g}", year, q)


PULLS = {
    "directory": pull_directory,
    "enrollment": pull_enrollment,
    "meps": pull_meps,
    "edfacts": pull_edfacts,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", choices=list(PULLS), default=None)
    args = ap.parse_args(argv)
    params = load_params()
    cfg = params["sources"]["urban_api"]
    todo = args.only or [k for k in PULLS
                         if k in ("directory", "enrollment")
                         or cfg.get(f"pull_{k}", False)]
    client = UrbanClient(params)
    for name in todo:
        log.info("pulling %s", name)
        PULLS[name](client, params)


if __name__ == "__main__":
    main()
