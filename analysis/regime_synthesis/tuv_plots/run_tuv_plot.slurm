#!/bin/bash
#SBATCH --job-name=tuv_plot
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs/tuv_plot_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs/tuv_plot_%j.err

mkdir -p /umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs

source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

echo "=== TUV Plot Analysis ==="
echo "Start: $(date)"
python -u /umbc/rs/pi_halem/users/yashd1/common/tuv_plots/plot_tuv_analysis.py
echo "Done: $(date)"
