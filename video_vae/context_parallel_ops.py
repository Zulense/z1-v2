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

    """
    This function creates a custom Pytorch Autograd function. It tells the neural network 
    exactly how to behave during the `forward pass` (processing data) and the `backward pass` (learning from errors)
    """
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_split(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        return _conv_gather(grad_output, ctx.dim, ctx.kernel_size), None, None


def _conv_gather(input_, dim=2, kernel_size=1):
    cp_world_size = get_context_parallel_world_size() # 2

    # Bypass the function if context parallel is 1
    if cp_world_size == 1:
        return input_

    group = get_context_parallel_group() 
    cp_rank = get_context_parallel_rank() # 0, 1 -> 1 is not working...

    # print('in _conv_gather, cp_rank:', cp_rank, 'input_size:', input_.shape) --> cp_rank: 0 input_size: torch.Size([2, 8, 3, 32, 32])

    ## it carefully isolates the overlapping boundary parts so they are accounted for correctly.
    # torch.Size([2, 8, 3, 32, 32]) -> torch.Size([3, 8, 2, 32, 32]) -> torch.Size([1, 8, 2, 32, 32]) -> torch.Size([2, 8, 1, 32, 32])
    input_first_kernel_ = input_.transpose(0, dim)[:kernel_size].transpose(0, dim).contiguous()

    # Worker 0 (if cp_rank == 0) has the very beginning of the movie.
    # Because they have the start of the movie, they do a different kind of trimming. In your code, Worker 0 actually extracts the first few frames into a separate variable called input_first_kernel_ and keeps it safe.
    if cp_rank == 0:
        # torch.Size([2, 8, 3, 32, 32]) -> torch.Size([3, 8, 2, 32, 32]) -> torch.Size([2, 8, 2, 32, 32]) -> torch.Size([2, 8, 2, 32, 32])
        input_ = input_.transpose(0, dim)[kernel_size:].transpose(0, dim).contiguous()

    # Worker 1 (else) has the middle/end of the movie.
    # Since Worker 0 already has the beginning of the movie perfectly sorted out, Worker 1's only job is to cut off the duplicate photocopied frames at the start of their chunk so it attaches perfectly to Worker 0's piece.
    else:
        # input_ = input_.transpose(0, dim)[max(kernel_size - 1, 0) :].transpose(0, dim).contiguous()
        assert("<-----------------[context_parallel_ops.py] [_conv_gather] i want to know that `Rank=1` is working or not. ----------------->")

    ## Rank 0 creates a list of two empty containers. It knows it needs one container for itself, and one container to catch the incoming data from GPU 1.
    # torch.Size([2, 8, 3, 32, 32]) + torch.Size([2, 8, 2, 32, 32]) -> [torch.Size([2, 8, 3, 32, 32]), torch.Size([2, 8, 2, 32, 32])]
    tensor_list = [torch.empty_like(torch.cat([input_first_kernel_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]
    

    if cp_rank == 0:
        # torch.Size([2, 8, 1, 32, 32]) + torch.Size([2, 8, 2, 32, 32]) -> torch.Size([2, 8, 3, 32, 32])
        input_ = torch.cat([input_first_kernel_, input_], dim=dim)

    # torch.Size([2, 8, 3, 32, 32]) = torch.Size([2, 8, 3, 32, 32])
    tensor_list[cp_rank] = input_
    ## When this line executes, Rank 0 and Rank 1 trade their data over the internal network connection:
    ## Rank 0 drops its 3 frames into the first spot of tensor_list.
    ## Rank 1 (running invisibly in the background) sends its 2 frames over the network, and PyTorch drops them into the second spot of Rank 0's tensor_list.
    ## So, after the all_gather line finishes, Rank 0's tensor_list looks like this:
    ## tensor_list[0] = [2, 8, 3, 32, 32] (Rank 0's own data)
    ## tensor_list[1] = [2, 8, 2, 32, 32] (The data PyTorch fetched from Rank 1)
    torch.distributed.all_gather(tensor_list, input_, group=group)

    ## Glues all the newly gathered pieces back together into one giant, continuous tensor.
    # [torch.Size([2, 8, 3, 32, 32]), torch.Size([2, 8, 2, 32, 32])] -> torch.Size([2, 8, 5, 32, 32])
    output = torch.cat(tensor_list, dim=dim).contiguous()

    # Let each GPU write its own little diary entry to a text file
    # with open(f"gpu_log_rank_{cp_rank}.txt", "a") as f:
    #     f.write(f"Hello from Rank {cp_rank}! My input was: {input_.shape}, and my output is: {output.shape}\n")
        
    # print('out _conv_gather, cp_rank:', cp_rank, 'input_size:', output.shape) -> cp_rank: 0 input_size: torch.Size([2, 8, 5, 32, 32])
    return output


def _conv_split(input_, dim=2, kernel_size=1):

    ## check how many GPUS are available, if there is only 1, it immediately returns the original data --> no slicing needed!
    cp_world_size = get_context_parallel_world_size()  # 2
    cp_rank = get_context_parallel_rank() 

    # Bypass the function if context parallel is 1
    if cp_world_size == 1:
        return input_

    # print('in _conv_split, cp_rank:', cp_rank, 'input_size:', input_.shape) # -> 0 , torch.Size([2, 3, 33, 256, 256])


    ## Calculate the base size of each slice. It takes the total size, subtracts the overlap (`kernel_size`) and divides by the number of GPUs.
    # torch.Size([2, 3, 33, 256, 256])[2] => (33 - 1) => 32 / 2 => 16
    dim_size = (input_.size()[dim] - kernel_size) // cp_world_size

    ## The very first GPU (Rank 0) grabs the first chunk of data, plus the extra overlapping (`kernel_size`)
    if cp_rank == 0:
        # torch.Size([2, 3, 33, 256, 256]) -> torch.Size([33, 3, 2, 256, 256]) -> torch.Size([17, 3, 2, 256, 256]) -> torch.Size([2, 3, 17, 256, 256])
        output = input_.transpose(dim, 0)[: dim_size + kernel_size].transpose(dim, 0)
    else:
        ## Every other GPU calculates it's specific start and end points 
        ## `.transpose()`: You will see this a lot here: it's a neat trick that temporarily rotates 
        # the tensor so the dimension we want to cut is at the very front, making the slicing math much easier to write.

        # output = input_.transpose(dim, 0)[cp_rank * dim_size + 1:(cp_rank + 1) * dim_size + kernel_size].transpose(dim, 0)
        # output = input_.transpose(dim, 0)[
        #     cp_rank * dim_size + kernel_size : (cp_rank + 1) * dim_size + kernel_size
        # ].transpose(dim, 0)
        assert("<-----------------[context_parallel_ops.py] [_conv_split] i want to know that `Rank=1` is working or not. ----------------->")


    output = output.contiguous()

    # print('out _conv_split, cp_rank:', cp_rank, 'input_size:', output.shape) -> 0 , torch.Size([2, 3, 17, 256, 256])

    return output


## <------------------------------------------------------------> ##
## <---- THIS CODE ARE WORK ON `vae.py` file ------------> ##
## <-------------------------------------------------------------> ##

def conv_gather_from_context_parallel_region(input_, dim, kernel_size):
    return _ConvolutionGatherFromContextParallelRegion.apply(input_, dim, kernel_size)

class _ConvolutionGatherFromContextParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_gather(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        return _conv_split(grad_output, ctx.dim, ctx.kernel_size), None, None