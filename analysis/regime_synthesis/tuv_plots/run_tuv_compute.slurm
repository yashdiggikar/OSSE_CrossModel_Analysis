#!/bin/bash
#SBATCH --job-name=tuv_compute
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs/tuv_compute_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs/tuv_compute_%j.err

mkdir -p /umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs
mkdir -p /umbc/rs/pi_halem/users/yashd1/common/tuv_plots/cache

source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

echo "=== TUV RMSE Compute ==="
echo "Start: $(date)"
python -u /umbc/rs/pi_halem/users/yashd1/common/tuv_plots/compute_tuv_rmse.py
echo "Done: $(date)"
