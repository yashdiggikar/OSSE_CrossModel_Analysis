#!/bin/bash
#SBATCH --cluster=chip-cpu
#SBATCH --job-name=vert_prop
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/logs_analysis/%x_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/logs_analysis/%x_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --account=pi_halem
#SBATCH --partition=general
#SBATCH --qos=medium

set -e
mkdir -p /umbc/rs/pi_halem/users/yashd1/logs_analysis
export HDF5_USE_FILE_LOCKING=FALSE
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
cd /umbc/rs/pi_halem/users/yashd1
echo "Running analyze_vertical_propagation.py on $(hostname), job ${SLURM_JOB_ID}"
python -u random/analyze_vertical_propagation.py
echo "Done: $(date)"
