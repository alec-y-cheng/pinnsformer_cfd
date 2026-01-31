#!/bin/bash
#SBATCH --job-name=pinnsformer_train
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H100:1                # GPU type (H100) and number of GPUs 
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out

module load anaconda3
conda activate myenv

cd ~/ondemand/PINNsformer
mkdir -p logs

nvidia-smi
which python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

export PYTORCH_ALLOC_CONF=expandable_segments:True
python train.py
