#!/bin/bash
#SBATCH --job-name=vert_prop_analysis
#SBATCH --account=pi_halem
#SBATCH --partition=gpu
#SBATCH --qos=chip-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/umbc/rs/pi_halem/users/yashd1/logs/vert_prop_%j.out
#SBATCH --error=/umbc/rs/pi_halem/users/yashd1/logs/vert_prop_%j.err

mkdir -p /umbc/rs/pi_halem/users/yashd1/logs
source /umbc/rs/pi_halem/users/yashd1/venvs/aurora310/bin/activate
python /umbc/rs/pi_halem/users/yashd1/random/analyze_vertical_propagation.py
