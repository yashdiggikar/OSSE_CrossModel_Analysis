#!/bin/bash
#SBATCH --job-name=perturbed_rmse
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/logs/perturbed_rmse_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/logs/perturbed_rmse_%j.err

export USER="${USER:-yashd1}"
RESEARCH_ROOT="/umbc/rs/pi_halem/users/${USER}"

export HOME="${RESEARCH_ROOT}"
export XDG_CACHE_HOME="${RESEARCH_ROOT}/.cache"
export XDG_CONFIG_HOME="${RESEARCH_ROOT}/.config"
export TMPDIR="${RESEARCH_ROOT}/.tmp/${SLURM_JOB_ID}"
export MPLCONFIGDIR="${RESEARCH_ROOT}/.config/matplotlib"
export HDF5_USE_FILE_LOCKING=FALSE

mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "${RESEARCH_ROOT}/logs"

source "${RESEARCH_ROOT}/venvs/aurora310/bin/activate"

echo "===================================="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Started : $(date)"
echo "Python  : $(which python)  $(python --version 2>&1)"
echo "===================================="

cd "${RESEARCH_ROOT}"
python -u random/compute_perturbed_rmse.py

echo "===================================="
echo "Finished: $(date)"
echo "===================================="
