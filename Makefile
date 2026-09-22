# Data prep pipeline, in order. Each step reads the previous step's outputs.
# Run from the repo root inside the conda env `main`.
PY = python -m

.PHONY: all study_area ccd panel_ids nlcd buffers smod panel diagnostics test

all: study_area ccd panel_ids nlcd buffers smod panel diagnostics

study_area:          ## TIGER counties -> data/interim/study_area/
	$(PY) src.ingest.counties

ccd:                 ## Urban portal pulls -> data/raw/urban/
	$(PY) src.ingest.ccd

panel_ids:           ## directory, ID crosswalk, sites and relocation flags
	$(PY) src.panel.directory
	$(PY) src.panel.crosswalk
	$(PY) src.panel.relocation

nlcd:                ## clip Annual NLCD impervious to the AOI (network heavy)
	$(PY) src.exposure.nlcd

buffers:             ## impervious fraction in 1/2/5 km buffers (run on HPC)
	$(PY) src.exposure.buffers

smod:                ## GHS-SMOD class and distance to urban clusters
	$(PY) src.exposure.smod

panel:               ## school-year analysis panel -> data/processed/
	$(PY) src.panel.build_panel

diagnostics:         ## fringe/treatment definitions and viability report
	$(PY) src.analysis.diagnostics

test:
	pytest -q
