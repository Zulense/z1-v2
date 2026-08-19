
# This scripts using 2 gpu to inference 
# Now only supports 2GPUs

GPUS=2

torchrun --nproc_per_node $GPUS \
    inference_multigpu.py