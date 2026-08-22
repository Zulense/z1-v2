import torch, argparse
from context_parallel import init_distributed_mode
from sequence_parallel import init_sequence_parallel_group

from video_vae.causal_conv import CausalConv3d


def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process Script', add_help=False)
    parser.add_argument('--sp_group_size', default=2, type=int, help='The number of GPUS used for inference, should 2 or 4')
    parser.add_argument('--model_dtype', default='bf16', type=str, help='The Model Dtype: bf16')

    return parser.parse_args()


def main():

    args = get_args()

    # setup DDP(Distributed Data Parallel)
    init_distributed_mode(args=args)
    assert args.world_size == args.sp_group_size, "The sequence parallel size should be DDP world size."


    # Enable sequence parallel 
    init_sequence_parallel_group(args)

    device = torch.device("cuda")
    rank = args.rank 
    model_dtype = args.model_dtype 

    model = CausalConv3d(input_channels=3,
                         output_chaannels=128,
                         kernel_size=3,
                         stride=1).to(device)

    x = torch.randn(32, 3, 8, 128, 128)
    out = model(x).to(device)
    print(out)
    

    
