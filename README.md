# school-urbanization

What happens to schools on the rural–urban fringe when urban expansion reaches
them. This repository holds the Atlanta pilot, which tests whether the design
works (enough treated schools, clean pre-periods, usable outcome series) before
scaling to other metros and countries. See [docs/overview.md](docs/overview.md)
for the design and the data decisions made so far.

Everything is Python. All thresholds, buffers, years and paths live in
[`config/params.yaml`](config/params.yaml).

## Setup

```bash
conda env update -n main -f environment.yml
conda activate main
pytest -q
```

## Pipeline

Run from the repo root. Each step reads the previous step's outputs, and every
file under `data/interim` and `data/processed` can be regenerated from these
scripts. `data/raw` is written once by the ingest steps and never modified.

| step | command | writes |
|---|---|---|
| study area | `python -m src.ingest.counties` | `data/interim/study_area/{counties,aoi}.gpkg` |
| school data | `python -m src.ingest.ccd` | `data/raw/urban/<pull>/<year>.parquet` |
| directory | `python -m src.panel.directory` | `data/interim/ccd/directory.parquet` |
| ID crosswalk | `python -m src.panel.crosswalk` | `data/interim/ccd/{id_crosswalk,school_uid}.parquet` |
| sites and relocations | `python -m src.panel.relocation` | `data/interim/ccd/{school_sites,sites}.parquet` |
| NLCD clip | `python -m src.exposure.nlcd` | `data/interim/nlcd/fctimp_<year>.tif` |
| buffer extraction | `python -m src.exposure.buffers` | `data/interim/exposure/impervious_buffers.parquet` |
| GHS-SMOD | `python -m src.exposure.smod` | `data/interim/exposure/smod_sites.parquet` |
| panel | `python -m src.panel.build_panel` | `data/processed/school_year_panel.parquet` |

`make all` runs them in order. `slurm/exposure.sbatch` runs the three raster
steps on the cluster.

## Key conventions

- School years are indexed by the fall year (2015 is 2015–16), following the Urban Institute portal.
- `ncessch` is the 12-character NCES ID. `school_uid` follows a school across NCES ID changes. `site_id` is a school at one location, so a relocated school gets a new site and a new exposure series.
- Closed and newly opened schools stay in the panel. Openings and closures are outcomes.
- Distances and buffers use NAD83 / UTM 16N (EPSG:26916). NLCD stays in its native EPSG:5070 grid and school buffers are reprojected onto it.
- Charter, virtual and non-regular schools are kept and flagged (`in_primary`), not dropped.

## Before the first real run

The sandbox this code was written in could not reach the data hosts, so the
ingest steps are tested on synthetic data only. Check these on the first run.

1. **Annual NLCD source.** `sources.nlcd.fctimp_template` is a best guess at the MRLC file naming. Point it at the current release, or at local copies of the CONUS files.
2. **Urban API field names.** The code expects `ncessch`, `county_code`, `grade_edfacts`, `*_test_pct_prof_midpt` and `meps*` columns. `python -m src.ingest.ccd --only directory` followed by a look at one year's parquet is a quick check.
3. **EDFacts year convention.** Confirm that Urban's `year` for assessments means the same school year as it does for CCD.
4. **County list.** `src.ingest.counties` fails if a FIPS code and name in `params.yaml` disagree with TIGER.
