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



## <------------------------------------------------------------> ##
## <---- THIS CODE ARE WORK ON `causal_conv.py` file ------------> ##
## <-------------------------------------------------------------> ##
def cp_pass_from_previous_rank(input_, dim, kernel_size):
    return _CPConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size)


class _CPConvolutionPassFromPreviousRank(torch.autograd.Function):

    ## when training  an AI, it has to move forward (making predictions) and backward (learning from mistakes).
    ## This block tells the AI exactly what to do in both directions
    ## it says, "Use `_cp_pass_from_previous_rank` to go forward and `_drop_from_previous_rank` to go backward." 
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _cp_pass_from_previous_rank(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size), None, None


def _cp_pass_from_previous_rank(input_, dim, kernel_size):

    # if the viewing window `kernel_size` is just 1, the worker does not need to 
    # look at anyone else's frames, so the code skips the rest of the steps and returns the video as-is.
    if kernel_size == 1:
        return input_

    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    cp_world_size = get_context_parallel_world_size()



    global_rank = torch.distributed.get_rank()

    ## it uses `.transpose()` to flip the data, moving the "time" dimenaion (the frames)
    ## to the very front. This makes it much easier to slice off the last few frames. 
    input_ = input_.transpose(0, dim)

    ## The worker does some quick math to figure out the ID of the person they need to send frames to (`send_rank`)
    ## and the person they need to receive frames from (`recv_rank`)
    send_rank = global_rank + 1
    recv_rank = global_rank - 1
    if send_rank % cp_world_size == 0:
        send_rank -= cp_world_size
    if recv_rank % cp_world_size == cp_world_size - 1:
        recv_rank += cp_world_size

    recv_buffer = torch.empty_like(input_[-kernel_size + 1 :]).contiguous()

    ## This is the mailroom. If the worker is not the last person in line, they mail out their last few frames (`isend`)
    ## if they are not the first person in line. they put out an empty box and wait to receive mail (`irecv`)
    if cp_rank < cp_world_size - 1:
        req_send = torch.distributed.isend(input_[-kernel_size + 1 :].contiguous(), send_rank, group=group)
    if cp_rank > 0:
        req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)

    ## Time to unpack. If this is the very first worker (`cp_rank==0`), they do not receive mail, so they just tape blank frames (zeros) to the front of their video. Everyone
    ## else waits to their package (`req_recv.wait()`) and glues those received frames to the start of their video.
    if cp_rank == 0:
        input_ = torch.cat([torch.zeros_like(input_[:1])] * (kernel_size - 1) + [input_], dim=0)
    else:
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)

    ## The data is flipped back to it's original shape and handed back to the main program.
    input_ = input_.transpose(0, dim).contiguous()
    return input_


## when the AI is learning (going backward). It cannot accidentally learn from it's neighbor's frames. 
## This function simply throws away those borrowed frames by slicing them off before the learning math happens.
def _drop_from_previous_rank(input_, dim, kernel_size):
    input_ = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim)
    return input_


## <------------------------------------------------------------> ##
## <---- THIS CODE ARE WORK ON `vae_wrapper.py` file ------------> ##
## <-------------------------------------------------------------> ##

def conv_scatter_to_context_parallel_region(input_, dim, kernel_size):
    return _ConvolutionScatterToContextParallelRegion.apply(input_, dim, kernel_size)

class _ConvolutionScatterToContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_split(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        return _conv_gather(grad_output, ctx.dim, ctx.kernel_size), None, None


def _conv_gather(input_, dim=2, kernel_size=1):
    cp_world_size = get_context_parallel_world_size()

    # Bypass the function if context parallel is 1
    if cp_world_size == 1:
        return input_

    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()

    # print('in _conv_gather, cp_rank:', cp_rank, 'input_size:', input_.shape)

    input_first_kernel_ = input_.transpose(0, dim)[:kernel_size].transpose(0, dim).contiguous()
    if cp_rank == 0:
        input_ = input_.transpose(0, dim)[kernel_size:].transpose(0, dim).contiguous()
    else:
        input_ = input_.transpose(0, dim)[max(kernel_size - 1, 0) :].transpose(0, dim).contiguous()

    tensor_list = [torch.empty_like(torch.cat([input_first_kernel_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]
    if cp_rank == 0:
        input_ = torch.cat([input_first_kernel_, input_], dim=dim)

    tensor_list[cp_rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    # print('out _conv_gather, cp_rank:', cp_rank, 'input_size:', output.shape)

    return output


def _conv_split(input_, dim=2, kernel_size=1):
    cp_world_size = get_context_parallel_world_size()

    # Bypass the function if context parallel is 1
    if cp_world_size == 1:
        return input_

    # print('in _conv_split, cp_rank:', cp_rank, 'input_size:', input_.shape)

    cp_rank = get_context_parallel_rank()

    dim_size = (input_.size()[dim] - kernel_size) // cp_world_size

    if cp_rank == 0:
        output = input_.transpose(dim, 0)[: dim_size + kernel_size].transpose(dim, 0)
    else:
        # output = input_.transpose(dim, 0)[cp_rank * dim_size + 1:(cp_rank + 1) * dim_size + kernel_size].transpose(dim, 0)
        output = input_.transpose(dim, 0)[
            cp_rank * dim_size + kernel_size : (cp_rank + 1) * dim_size + kernel_size
        ].transpose(dim, 0)
    output = output.contiguous()

    # print('out _conv_split, cp_rank:', cp_rank, 'input_size:', output.shape)

    return output


