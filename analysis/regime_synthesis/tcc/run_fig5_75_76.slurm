#!/bin/bash
#SBATCH --job-name=tcc_t_ins_4m
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/common/tcc/logs/tcc_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/common/tcc/logs/tcc_%j.err
mkdir -p /umbc/rs/pi_halem/users/yashd1/common/tcc/logs
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
python /umbc/rs/pi_halem/users/yashd1/common/tcc/fig5_75_76_temp_ins_cloud_regime.py
