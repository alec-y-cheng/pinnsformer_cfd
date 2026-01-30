#!/bin/bash
#SBATCH --job-name=convert_npz        # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --mem=64G                      # Memory allocation
#SBATCH --time=01:00:00               # Walltime (adjust as needed)
#SBATCH --output=logs/convert_%j.out  # Log file

# Load Python or activate your environment
module load python/3.12
# Or if using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

# Navigate to your repo folder
cd /home/your_username/pinnsformer/

# Make sure the logs folder exists
mkdir -p logs

# Run the conversion
python convert_npz.py
