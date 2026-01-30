#!/bin/bash
#SBATCH --job-name=convert_npz        # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --mem=64G                      # Memory allocation
#SBATCH --time=01:00:00               # Walltime (adjust as needed)
#SBATCH --output=logs/convert_%j.out  # Log file

# Load Anaconda module
module load anaconda3

# Activate your pinnsformer environment

# Navigate to your actual repo folder
cd /home/hice1/acheng324/ondemand/PINNsformer/

# Make sure the logs folder exists
mkdir -p logs

# Run the conversion
python convert_npz.py
