import torch, argparse
from context_parallel import init_distributed_mode


def get_args():
    parser = argparse.ArgumentParser('Pytorch Multi-process Script', add_help=False)
    parser.add_argument('--sp_group_size', default=2, type=int, help='The number of GPUS used for inference, should 2 or 4')

    return parser.parse_args()


def main():

    args = get_args()

    # setup DDP(Distributed Data Parallel)
    init_distributed_mode(args=args)
    assert args.world_size == args.sp_group_size, "The sequence parallel size should be DDP world size."

    
