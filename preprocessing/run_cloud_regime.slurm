#!/bin/bash
#SBATCH --job-name=cloud_regime_5_12
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/logs/cloud_regime_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/logs/cloud_regime_%j.err

mkdir -p /umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/logs

source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

python /umbc/rs/pi_halem/users/yashd1/ALL_ERA5/tcc/analyze_cloud_regime.py
