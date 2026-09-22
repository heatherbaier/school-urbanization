#!/bin/bash
# NLCD clipping and buffer extraction on the ASU cluster.
#   sbatch slurm/exposure.sbatch
# Assumes study_area and panel_ids have already been run (sites.parquet exists).
#SBATCH --job-name=su-exposure
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

python -m src.exposure.nlcd
python -m src.exposure.buffers
python -m src.exposure.smod
