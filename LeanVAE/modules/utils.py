import torch, math, imageio
import torch.distributed as dist
import logging
import numpy as np

def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""

    if dist.get_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handles=[logging.StreamHandler(),
                     logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        
    else:   # dummy logger (does nothing.)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler)

    return logger


def save_video_grid(video, fname, nrow=None, fps=3):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1).contiguous()

    video = (video.detach().cpu().numpy() * 255).astype('unit8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)

    padding = 0
    video_grid = np.zeros((t, (padding+h) * nrow + padding,
                           (padding+w) * ncol+padding, c),
                           dtype='unit8')

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding+h) * r
        start_c = (padding+w) * c
        video_grid[:, start_r:start_r+h, start_c:start_c+w] = video[i]

    video = []
    for i in range(t):
        video.append(video_grid[i])
    imageio.mimsave(fname, video, fps=fps)
    
