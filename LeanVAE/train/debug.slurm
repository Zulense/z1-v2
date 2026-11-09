#!/bin/bash
#SBATCH --job-name=multi_gpu_inference
#SBATCH --partition=dgxnp        # The DGX GPU partition we found from sinfo
#SBATCH --nodes=1                # We only need 1 DGX node
#SBATCH --gpus-per-node=2        # Request all 8 A100 GPUs on that node
#SBATCH --cpus-per-task=4       # Request CPU cores to feed data to the GPUs
#SBATCH --time=01:00:00          # 2 hours max (well under the 7-day limit)
#SBATCH --output=logs/inference_%j.log # Save the output to a log file



# Define the number of GPUs
GPUS=2

# Run the inference script
torchrun --nproc-per-node=$GPUS \
    train.py \
    --default_root_dir "../LeanVAE_ex/run1" \
    --num_nodes 1 \
    --gpus 8 \
    --grad_clip_val 1.0 \
    --lr 5e-5 \
    --lr_min 1e-5 \
    --warmup_steps 5000 \
    --discriminator_iter_start 600000 \
    --progress_bar_refresh_rate 500 \
    --max_step 700000 \
    --data_path '' \
    --train_datalist "../kinetics-dataset/train/" \
    --val_datalist "../kinetics-dataset/valid/" \
    --batch_size 5 \
    --num_workers 12 \
    --sample_rate 1 \
    --sequence_length 17 \
    --latent_dim 4 \
    --ista_iter_num 2 \
    --ista_layer_num 2 \
    --l_dim 128 \
    --h_dim 384 \
    --sep_num_layer 2 \
    --fusion_num_layer 4 \

