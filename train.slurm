#!/bin/bash
#SBATCH --job-name=pinnsformer_train
#SBATCH --partition=pace-gpu
#SBATCH --gres=gpu:H100:1                # GPU type (H100) and number of GPUs 
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out

module load anaconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pinnsformer

cd ~/ondemand/PINNsformer

python train_pinnsformer.py
