#!/bin/bash
#SBATCH --job-name=q_nrmse_3models
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/common/logs/q_nrmse_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/common/logs/q_nrmse_%j.err

mkdir -p /umbc/rs/pi_halem/users/yashd1/common/logs

source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

echo "=== FCN Q/RH500 NRMSE ==="
python -u /umbc/rs/pi_halem/users/yashd1/FourCastNet/nrmse/compute_q_nrmse_fcn.py

echo "=== Aurora Q500 NRMSE ==="
python -u /umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files/nrmse/compute_q_nrmse_aurora.py

echo "=== GraphCast Q500 NRMSE ==="
python -u /umbc/rs/pi_halem/users/yashd1/gc_pred_files/nrmse/compute_q_nrmse_gc.py

echo "=== ALL DONE ==="
