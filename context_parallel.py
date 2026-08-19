from torch import distributed as dist
import os, torch 

_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_SIZE = None


def is_distribution_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True



def get_rank():
    if not is_distribution_avail_and_initialized():
        return 0 

    return dist.get_rank()


def get_context_parallel_rank():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel rank is not recognized."

    rank = get_rank()
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE
    return cp_rank

def get_context_parallel_group_rank():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel group rank is not initialized."

    rank = get_rank()
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE

    return cp_group_rank

def get_context_parallel_world_size():
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel world size is not initialized."
    return _CONTEXT_PARALLEL_SIZE

def get_context_parallel_group():
    assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized."

    return _CONTEXT_PARALLEL_GROUP




# -----
def setup_for_distributed(is_master):
    """This function disables printing when not in master process"""

    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



def init_distributed_mode(args, init_pytorch_ddp=True):

    if int(os.getenv('OMPI_COMM_WORLD_SIZE', '0')) > 0:
        rank = int(os.environ('OMPI_COMM_WORLD_RANK'))
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

        os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ["RANK"] = os.environ['OMPI_COMM_WORLD_RANK']
        os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])


    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])

    else:
        print('Not using distributed mode')
        args.distributed = False 
        return 


    args.distributed = True
    args.dist_backend = 'nccl'
    args.dist_url = "env://"
    print(f'| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}', flush=True)

    if init_pytorch_ddp:

        # init ddp group, for script with using accelerate framework.
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend=args.dist_backend, 
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=args.rank,
                                             )
        torch.distributed.barrier()
        setup_for_distributed(is_master=args.rank==0)




