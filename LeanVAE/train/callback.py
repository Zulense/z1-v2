import torch, os
from pytorch_lightning.callbacks import Callback
import numpy as np
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from modules.utils import save_video_grid

class VideoLogger(Callback):

    def __init__(self,
                 batch_frequency,
                 max_videos,
                 clamp=True,
                 increase_log_steps=True):
        
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_videos = max_videos
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) +1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def log_local(self,
                    save_dir,
                    split,
                    videos,
                    global_step,
                    current_epoch,
                    batch_idx):

        root = os.path.join(save_dir, "videos", split)
        for k in videos:
            grid = videos[k] + 0.5
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.mp4".format(global_step, current_epoch, batch_idx, k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            save_video_grid(grid, path)


    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass

            return True
        return False
    

    def log_vid(self, pl_module, batch, batch_idx, split="train"):

        if (self.check_frequency(batch_idx) and 
            hasattr(pl_module, "log_videos") and callable(pl_module.log_videos) and self.max_videos > 0):

            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                videos = pl_module.log_videos(batch, split=split, batch_idx=batch_idx)

            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().cpu()
                    if self.clamp:
                        videos[k] = torch.clamp(videos[k], -0.5, 0.5)

            self.log_local(save_dir=pl_module.logger.save_dir,
                           split=split,
                           videos=videos,
                           global_step=pl_module.global_step,
                           current_epoch=pl_module.current_epoch,
                           batch_idx=batch_idx)

            if is_train:
                pl_module.train()


    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass 
            return True
        return False


    def on_train_batch_end(self,
                           trainer,
                           pl_module,
                           outputs,
                           batch,
                           batch_idx,
                           dataloader_idx):

        if batch[0]['video'].ndim == 4:
            return
        self.log_vid(pl_module, batch, batch_idx, split="train")


    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx):
        self.log_vid(pl_module=pl_module, batch=batch, batch_idx=batch_idx, split="val")


    






