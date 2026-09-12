import torch 

import sys 
from pathlib import Path
# Add the parent directory (z1-v2)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from context_parallel import (
    get_context_parallel_group, 
    get_context_parallel_rank, 
    get_context_parallel_group_rank, 
    get_context_parallel_world_size)



def cp_pass_from_previous_rank(input_, dim, kernel_size):
    return _CPConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size)


class _CPConvolutionPassFromPreviousRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _cp_pass_from_previous_rank(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size), None, None


def _cp_pass_from_previous_rank(input_, dim, kernel_size):
    # Bypass the function if kernel size is 1
    if kernel_size == 1:
        return input_

    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    cp_group_rank = get_context_parallel_group_rank()
    cp_world_size = get_context_parallel_world_size()

    # print('in _pass_from_previous_rank, cp_rank:', cp_rank, 'input_size:', input_.shape)

    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()

    input_ = input_.transpose(0, dim)

    # pass from last rank
    send_rank = global_rank + 1
    recv_rank = global_rank - 1
    if send_rank % cp_world_size == 0:
        send_rank -= cp_world_size
    if recv_rank % cp_world_size == cp_world_size - 1:
        recv_rank += cp_world_size

    recv_buffer = torch.empty_like(input_[-kernel_size + 1 :]).contiguous()
    if cp_rank < cp_world_size - 1:
        req_send = torch.distributed.isend(input_[-kernel_size + 1 :].contiguous(), send_rank, group=group)
    if cp_rank > 0:
        req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)

    if cp_rank == 0:
        input_ = torch.cat([torch.zeros_like(input_[:1])] * (kernel_size - 1) + [input_], dim=0)
    else:
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)

    input_ = input_.transpose(0, dim).contiguous()
    return input_


def _drop_from_previous_rank(input_, dim, kernel_size):
    input_ = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim)
    return input_