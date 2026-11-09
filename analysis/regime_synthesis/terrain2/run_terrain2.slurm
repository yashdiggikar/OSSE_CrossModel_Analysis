#!/bin/bash
#SBATCH --job-name=terrain2_uv_4m
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/common/terrain2/logs/terrain2_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/common/terrain2/logs/terrain2_%j.err
mkdir -p /umbc/rs/pi_halem/users/yashd1/common/terrain2/logs
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
python /umbc/rs/pi_halem/users/yashd1/common/terrain2/fig_terrain2_uv_ins.py
