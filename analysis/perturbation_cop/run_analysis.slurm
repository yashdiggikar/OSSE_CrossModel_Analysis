#!/bin/bash
#SBATCH --cluster=chip-cpu
#SBATCH --job-name=perturb_analysis
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/logs/%x_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/logs/%x_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --account=pi_halem
#SBATCH --partition=general
#SBATCH --qos=medium

# Generic CPU-only analysis-script runner for PERTURBATION_EXPERIMENTS.
# Usage: sbatch run_analysis.slurm <model_dir_name> <script_name.py>
# e.g.:  sbatch run_analysis.slurm FCN compute_global_rmse_corr.py

set -e
mkdir -p /umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/logs

MODEL_DIR="$1"
SCRIPT="$2"

# HDF5's file-locking is known to conflict with itself over NFS when multiple
# processes/nodes open the same large file concurrently (exactly the pattern
# of submitting all of a model's analysis jobs at once against one shared
# control file) -- disabling it is the standard workaround.
export HDF5_USE_FILE_LOCKING=FALSE

source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
cd "/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS/${MODEL_DIR}"
echo "Running ${SCRIPT} in ${MODEL_DIR} on $(hostname), job ${SLURM_JOB_ID}"
python -u "${SCRIPT}"
echo "Done: $(date)"
