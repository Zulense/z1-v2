import torch 
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
import time

class IterLoader:

    """
    A wrapper to convert DataLoader as an infinite iterator.
    """

    def __init__(self,
                 dataloader: DataLoader,
                 use_distributed: bool = False,
                 epoch: int = 0):

        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._use_distributed = use_distributed
        self._epoch = epoch


    @property
    def epoch(self) -> int:
        return self._epoch


    def __next__(self):
        try:
            data = next(self.iter_loader)

        except StopIteration:
            self._epoch += 1 
            if hasattr(self._dataloader.sample, "set_epoch") and self._use_distributed:
                self._dataloader.sample.set_epoch(self._epoch)
            time.sleep(2)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data 

    def __iter__(self):
        return self 

    def __len__(self):
        return len(self._dataloader)



def video_dataloaders(dataset,
                      batch_size,
                      num_workers,
                      world_size=None,
                      rank=None,
                      epoch=0,
                      ):

    """The video training dataloader builder"""

    assert world_size is not None
    assert rank is not None

    # only use video data 
    video_gpus = world_size

    sampler = DistributedSampler(
        dataset=dataset,
        shuffle=True,
        num_replicas=video_gpus,
        rank=rank,
        seed=epoch
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        collate_fn=default_collate,
        drop_last=True
    )

    loader = IterLoader(dataloader=loader,
                        use_distributed=True,
                        epoch=epoch)

    return loader


