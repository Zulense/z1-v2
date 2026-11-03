#!/bin/bash
#SBATCH --job-name=multi_gpu_inference
#SBATCH --partition=dgxnp        # The DGX GPU partition we found from sinfo
#SBATCH --nodes=1                # We only need 1 DGX node
#SBATCH --gpus-per-node=1        # Request all 8 A100 GPUs on that node
#SBATCH --cpus-per-task=4       # Request CPU cores to feed data to the GPUs
#SBATCH --time=01:00:00          # 2 hours max (well under the 7-day limit)
#SBATCH --output=logs/inference_%j.log # Save the output to a log file



# Define the number of GPUs
GPUS=1

# Run the inference script
torchrun --nproc-per-node=$GPUS \
    debug.py \
    --sp_group_size=$GPUS

EOF