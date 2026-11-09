#!/bin/bash
#SBATCH --job-name=graphcast-tuv-synergy-cache
#SBATCH --partition=gpu
#SBATCH --clusters=chip-gpu
#SBATCH --qos=chip-gpu
#SBATCH --constraint="L40S"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs/graphcast_tuv_cache_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs/graphcast_tuv_cache_%j.err

set -euo pipefail

BASE=/umbc/rs/pi_halem/users/yashd1/Graphcast-full
module purge
module load Python/3.13.5-GCCcore-14.3.0
source "$BASE/venv_graphcast/bin/activate"

mkdir -p /umbc/rs/pi_halem/users/yashd1/common/tuv_plots/logs
export MPLCONFIGDIR="/umbc/rs/pi_halem/users/yashd1/common/tuv_plots/.mplcache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

cd /umbc/rs/pi_halem/users/yashd1/common/tuv_plots
python compute_graphcast_tuv_cache.py

echo "DONE"
