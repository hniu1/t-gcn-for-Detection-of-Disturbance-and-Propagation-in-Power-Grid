#!/bin/bash
#SBATCH -A cli138
#SBATCH -J tgcn_train
#SBATCH -o logs/tgcn_train-%j.out
#SBATCH -e logs/tgcn_train-%j.err
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00

set -euo pipefail

source /ccs/home/haoranniu/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/orion/proj-shared/cli138/7hn/envs/tnvr

export PYTHONIOENCODING=utf-8

workspace_root="/lustre/orion/proj-shared/cli138/7hn/FNET"
project_root="$workspace_root/t-gcn-for-Detection-of-Disturbance-and-Propagation-in-Power-Grid"

mkdir -p "$project_root/logs"
cd "$project_root"

data_dir="$workspace_root/data/2024-06-01"
metadata_file="$workspace_root/data/FDRLocation.xlsx"
adjacency_file="$project_root/results/A_geo.npy"
inferred_locs_file="$project_root/results/inferred_locations.csv"

output_dir="$project_root/results_masked_cov80_full_scaled"

tin=100
horizon=10
hidden_dim=64
num_layers=2
lr=1e-3
batch_size=128
epochs=50

stride_train=10
stride_val=10
stride_test=10
stride_pred=10

train_frac=0.80
val_frac=0.05
test_frac=0.05
pred_frac=0.10

min_step_coverage=0.8
short_gap_steps=10
medium_gap_steps=300
sample_rate_hz=10.0
feature_scaling=robust
scaled_clip_value=10.0

echo "Starting masked full-data T-GCN training"
echo "Project root: $project_root"
echo "Output dir:   $output_dir"
echo "Coverage min: $min_step_coverage"
echo "Strides:      train=$stride_train val=$stride_val test=$stride_test pred=$stride_pred"
echo "Splits:       train=$train_frac val=$val_frac test=$test_frac pred=$pred_frac"
echo "Scaling:      $feature_scaling (clip=$scaled_clip_value)"
echo "Epochs:       $epochs"
echo "Batch size:   $batch_size"

srun python -u train.py \
	--data_dir "$data_dir" \
	--metadata_file "$metadata_file" \
	--adjacency_file "$adjacency_file" \
	--inferred_locs_file "$inferred_locs_file" \
	--output_dir "$output_dir" \
	--Tin "$tin" \
	--H "$horizon" \
	--hidden_dim "$hidden_dim" \
	--num_layers "$num_layers" \
	--lr "$lr" \
	--batch_size "$batch_size" \
	--epochs "$epochs" \
	--train_frac "$train_frac" \
	--val_frac "$val_frac" \
	--test_frac "$test_frac" \
	--pred_frac "$pred_frac" \
	--stride_train "$stride_train" \
	--stride_val "$stride_val" \
	--stride_test "$stride_test" \
	--stride_pred "$stride_pred" \
	--min_step_coverage "$min_step_coverage" \
	--short_gap_steps "$short_gap_steps" \
	--medium_gap_steps "$medium_gap_steps" \
	--sample_rate_hz "$sample_rate_hz" \
	--feature_scaling "$feature_scaling" \
	--scaled_clip_value "$scaled_clip_value" \
	--device cuda