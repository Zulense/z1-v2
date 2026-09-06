#!/bin/bash
#SBATCH --job-name=train_video_vae
#SBATCH --partition=dgxnp        # The DGX GPU partition we found from sinfo
#SBATCH --nodes=1                # We only need 1 DGX node
#SBATCH --gpus-per-node=2        # Request all 8 A100 GPUs on that node
#SBATCH --cpus-per-task=4       # Request CPU cores to feed data to the GPUs
#SBATCH --time=01:00:00          # 2 hours max (well under the 7-day limit)
#SBATCH --output=vae_logs/inference_%j.log # Save the output to a log file




# Define the number of GPUs
GPUS=2
VIDEO_ANNO=/nlsasfs/home/hfgenai/mansav/z1-v2/annotation/class_5_math_white_board.jsonl
RESOLUTION=256
NUM_FRAMES=33   # 13 * context_size - 1 
BATCH_SIZE=2
CONTEXT_SIZE=2   # context parallel size, GPUS % CONTEXT_SIZE=0
LPIPS_CKPT=/nlsasfs/home/hfgenai/mansav/z1-v2/PATH/vgg_lpips.pth


# Run the inference script, making sure sp_group_size matches world_size
torchrun --nproc-per-node=$GPUS \
    train/train_video_vae.py \
    --max_frames=$NUM_FRAMES \
    --batch_size=$BATCH_SIZE \
    --num_workers 6 \
    --seed 42 \
    --context_size $CONTEXT_SIZE \
    --video_anno=$VIDEO_ANNO \
    --batch_size=$BATCH_SIZE \
    --model_dtype bf16 \
    --use_context_parallel \
    --context_size $CONTEXT_SIZE \
    --lpips_ckpt=$LPIPS_CKPT \
    --resolution=$RESOLUTION \
    --disc_start 25000 \
    --kl_weight 1e-12 \
    --pixelloss_weight 10.0 \
    --perceptual_weight 1.0 \
    --disc_weight 0.5 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --seed 42 \
    --weight_decay 1e-3 \
    --clip_grad 1.0 \
    --lr 1e-4 \
    --lr_disc 1e-4 \
    --warmup_epochs 1 \
    --epochs 100 \
    --iters_per_epoch 20000 \
    --print_freq 40 \
    

    



