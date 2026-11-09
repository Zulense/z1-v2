from utils import is_dist_avail_and_initialized
import torch 

SEQ_PARALLEL_GROUP = None
SEQ_PARALLEL_SIZE = None
SEQ_PARALLEL_PROC_NUM = None   # using how many process for sequence parallel 

SYNC_INPUT_GROUP = None
SYNC_INPUT_SIZE = None





def init_sequence_parallel_group(args):
    global SEQ_PARALLEL_GROUP
    global SEQ_PARALLEL_SIZE
    global SEQ_PARALLEL_PROC_NUM

    assert SEQ_PARALLEL_GROUP is None, "sequence parallel group is already initialized."
    assert is_dist_avail_and_initialized(), "The Pytorch distributed should be initialized."
    SEQ_PARALLEL_SIZE = args.sp_group_size

    print(f"Setting the sequence Parallel Size {SEQ_PARALLEL_SIZE}")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if args.sp_proc_num == -1:
        SEQ_PARALLEL_PROC_NUM = world_size
    else:
        SEQ_PARALLEL_PROC_NUM = args.sp_proc_num 

    assert SEQ_PARALLEL_PROC_NUM % SEQ_PARALLEL_SIZE == 0, "The Process needs to be evenly divided."

    for i in range(0, SEQ_PARALLEL_PROC_NUM, SEQ_PARALLEL_SIZE):
        ranks = list(range(i, i + SEQ_PARALLEL_SIZE))
        groups = torch.distributed.new_group(ranks)
        if rank in ranks:
            SEQ_PARALLEL_GROUP = groups
            break

        


