#!/bin/bash
#SBATCH -A cli138
#SBATCH -J tgcn_train
#SBATCH -o logs/tgcn_train-%j.out
#SBATCH -e logs/tgcn_train-%j.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00

set -e

source /ccs/home/haoranniu/miniconda3/bin/activate base
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/tnvr

cd /lustre/orion/proj-shared/cli138/7hn/FNET/t-gcn-for-Detection-of-Disturbance-and-Propagation-in-Power-Grid

# Manually set these paths before submitting if needed
data_dir="/lustre/orion/proj-shared/cli138/7hn/FNET/data/2024-06-01"
metadata_file="/lustre/orion/proj-shared/cli138/7hn/FNET/data/FDRLocation.xlsx"

python -u train.py \
	--data_dir "$data_dir" \
	--metadata_file "$metadata_file" \
	--device cuda