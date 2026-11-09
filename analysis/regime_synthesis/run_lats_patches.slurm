#!/bin/bash
#SBATCH --job-name=lats_patch
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=1-4
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/common/logs/lats_patch_%A_%a.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/common/logs/lats_patch_%A_%a.err

mkdir -p /umbc/rs/pi_halem/users/yashd1/common/logs

echo "========================================"
echo "Job: $SLURM_JOB_ID  Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME  Started: $(date)"
echo "========================================"

source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate

case $SLURM_ARRAY_TASK_ID in
  1)
    echo "--- Aurora v500 latband + latslice patch ---"
    python3 /umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files/nrmse/patch_v500_aurora_lats.py
    ;;
  2)
    echo "--- FourCastNet v500 latband + latslice patch ---"
    python3 /umbc/rs/pi_halem/users/yashd1/FourCastNet/nrmse/patch_v500_fcn_lats.py
    ;;
  3)
    echo "--- GraphCast-S v500 latband + latslice patch ---"
    python3 /umbc/rs/pi_halem/users/yashd1/gc_pred_files/nrmse/patch_v500_gc_lats.py
    ;;
  4)
    echo "--- Prithvi U_lev51 + V_lev51 latband + latslice patch ---"
    python3 /umbc/rs/pi_halem/users/yashd1/prithvi_new/nrmse/patch_uv_lev51_prithvi_lats.py
    ;;
esac

echo "Finished: $(date)"
