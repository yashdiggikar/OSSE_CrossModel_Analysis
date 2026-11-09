#!/bin/bash
#SBATCH --cluster=chip-cpu
#SBATCH --job-name=tuv_rmse
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs/%x_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs/%x_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --account=pi_halem
#SBATCH --partition=general
#SBATCH --qos=medium

set -e
mkdir -p /umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs
export HDF5_USE_FILE_LOCKING=FALSE
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
cd /umbc/rs/pi_halem/users/yashd1/common/tuv_plots
echo "Running compute_tuv_rmse.py on $(hostname), job ${SLURM_JOB_ID}"
python -u compute_tuv_rmse.py
echo "Done: $(date)"
