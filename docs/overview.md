# Peri-urban schools and urban expansion — Atlanta pilot

## Project summary

This project asks what happens to schools on the rural–urban fringe when urban
expansion reaches and envelops them. The full paper compares the US, Brazil and
one European country (England or Portugal) using 15 to 20 years of school
administrative data linked to annual land cover and population grids. This
repository is the Atlanta pilot. It tests whether the design works, meaning
enough treated schools, clean pre-periods and usable outcome series, before
scaling to other metros and countries.

Urbanicity is treated as a process rather than a fixed category. Education
research almost always uses static rural and urban labels, so schools in
transition are misclassified or invisible. The comparative hypothesis is that
the same physical process produces different educational signatures under
different urbanization regimes. US sprawl tends to bring in more affluent
families, Brazilian peripheral expansion tends to bring in lower-income
families, and European planned growth should show muted change.

Outcome changes will mostly reflect sorting of new families into the area
rather than school effects. The analysis separates composition from
performance wherever possible and never implies a school effect from level
scores alone.

## Pilot scope

The study area is the 29-county Atlanta–Sandy Springs–Roswell MSA (2013
delineation), listed in `config/params.yaml`. The outer ring (Forsyth,
Cherokee, Paulding, Henry, Barrow, Walton and neighbours) holds most of the
2000s growth. School data runs from 2000 onward. Traditional public schools
are the primary sample. Georgia's charter sector is smaller than Arizona's, but
charters are still flagged and kept out of the primary sample.

## Data decisions so far

**SEDA is not used for school-level outcomes.** SEDA publishes school
estimates pooled across years (an average, a grade slope and a cohort trend),
not a school-by-year series. That cannot support a school-level event study.
SEDA's year-specific estimates exist only for districts and larger units.

**The Urban Institute Education Data Portal is the main school source.** It
provides the following school-by-year series.

- CCD directory, with coordinates, school type, status, charter flag, teachers, and free and reduced-price lunch counts
- CCD enrollment by grade and by race
- MEPS (Model Estimates of Poverty in Schools), comparable across states and years, covering 2009–10 through 2022–23. This is the economic disadvantage measure, since free lunch counts break after the Community Eligibility Provision
- EDFacts proficiency rates by grade and subject, roughly 2009–2018 plus later years. Georgia moved from the CRCT to Georgia Milestones in 2014–15, so these are standardized within year, grade and subject statewide and read only as relative position

Georgia's own GOSA files are a possible secondary achievement source with
finer detail.

**Race categories are harmonized.** Asian and Pacific Islander are merged, and
the multiracial category only exists from about 2009–10. The break-free
diversity indices use five groups reported in every year.

**Land and population exposure.** Annual NLCD fractional impervious surface is
the primary land-based exposure, measured in 1, 2 and 5 km buffers around each
school site. GHS-SMOD gives the harmonized Degree of Urbanisation class in
five-year epochs, with 2025 and 2030 marked as projections.

## Definitions

All thresholds are in `config/params.yaml` under `definitions`. The fringe
sample requires a low impervious fraction in the 2 km buffer at baseline and a
location within 10 km of a GHS-SMOD urban cluster. The land-based treatment
event is the first year the buffer crosses the impervious threshold and stays
above it for at least three years. The baseline is each school's first
observed year by default, since many fringe schools in the outer counties
opened after 2000.

## Status

Written and tested on synthetic data.

1. CCD directory, enrollment, MEPS and EDFacts pulls, the ID crosswalk and relocation flags
2. Annual NLCD clipping
3. Buffer extraction per site and year
4. GHS-SMOD class and distance to urban clusters per site and epoch
5. The school-year panel

Next come the fringe and treatment definitions with the diagnostic report
(step 5 of the plan), then the maps and a first Callaway–Sant'Anna event
study in Python.

## Open decisions

- Final treatment threshold and baseline fringe cutoff, after inspecting the Atlanta distributions
- Whether five-year GHS-SMOD epochs are fine enough or annual WorldPop-based classes are needed
- How charter schools enter the analysis beyond being flagged
- Which European country joins the full paper and which Brazilian metro serves as its pilot
