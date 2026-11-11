import torch, os, json
import pytorch_lightning as pl
import torch.utils.data as data
from decord import VideoReader
import random
import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Optional, List

from data_utils import VideoNorm
from dataset_func_utils import get_length_grouped_indices







class LengthGroupedSampler(Sampler):
    """
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while.
    keeping a bit of randomness.
    """

    def __init__(self,
                 batch_size: int,
                 world_size: int,
                 gradient_accumulation_size: int = 1,
                 initial_global_step: int = 0,
                 lengths: Optional[List[int]] = None,
                 group_data=False,
                 generator=None,
                 rank: Optional[int] = None,
                 seed: int = 0):

        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.initial_global_step = initial_global_step
        self.lengths = lengths
        self.group_data = group_data
        self.generator = generator
        self.gradient_accumulation_size = gradient_accumulation_size

        self.rank = rank
        self.epoch = 0
        self.seed = seed

        megabatch_size = self.batch_size * self.world_size
        self.num_samples = ((len(lengths) + megabatch_size-1) // megabatch_size) * self.batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        megabatch_indices = get_length_grouped_indices(lengths=self.lengths,
                                                       batch_size=self.batch_size,
                                                       world_size=self.world_size,
                                                       gradient_accumulation_size=self.gradient_accumulation_size,
                                                       initial_global_step=self.initial_global_step,
                                                       group_data=self.group_data,
                                                       generator=g)

        # subsample
        indices = [i for batch in megabatch_indices[self.rank::self.world_size] for i in batch]
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch






class MultiSizeVideoDataset(data.Dataset):

    """
    A flexible dataset for loading videos of different resolutions stored in a structured format.
    This dataset reads video file path from text file, 
    where each file corresponds to a specific resolution. 
    Returns: BCTHW videos in range [-0.5, 0.5]
    """

    def __init__(self, 
                 data_list,
                 data_folder=None,
                 sequence_length=17,
                 train=True,
                 sample_rate=1,
                 dynamic_sample=False):

        super().__init__()
        self.train = train
        self.data_folder = data_folder
        self.sequence_length = sequence_length
        self.dynamic_sample = dynamic_sample
        self.sample_rate = sample_rate

        self.lengths = []
        self.annotations = []
        # for dir in os.listdir(data_list):
        #     file_path = os.path.join(data_list, dir)
        #     with open(file_path) as f:
        #         anno = [anno.strip() for anno in f.readlines()]
        #         self.annotations.extend(anno)
        #         self.lengths.extend([dir] * len(anno))

        ## case 1: data_list is a .jsonl file ##
        if os.path.isfile(data_list) and data_list.endwith('.jsonl'):
            with open(data_list, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    raw_video_path = item["video"]

                    if self.data_folder is not None:
                        candidate_1 = os.path.join(self.data_folder, os.path.basename(raw_video_path))
                        candidate_2 = os.path.join(self.data_folder, raw_video_path.lstrip("/"))

                        if os.path.exists(candidate_1):
                            video_path = candidate_1
                        elif os.path.exists(candidate_2):
                            video_path = candidate_2
                        else:
                            assert FileNotFoundError
                    else:
                        video_path = raw_video_path

                    item["video"] = video_path
                    self.annotations.append(item)

                    self.lengths.append(str(item.get("frame", sequence_length)))



        self.norm = VideoNorm()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        # video_path = self.annotations[idx] if self.data_folder is None else os.path.join(self.data_folder, self.annotations[idx])

        ann = self.annotations[idx]
        if isinstance(ann, dict):
            video_path = ann["video"]
            caption = ann.get("caption", "")
        else:
            video_path = ann if self.data_folder is None else os.path.join(self.data_folder, ann.lstrip("/"))
            caption = ""



        try:
            decord_vr = VideoReader(video_path)
            total_frames = len(decord_vr)

        except Exception as e:
            raise RuntimeError(f"Failed to read video: {video_path}. Error: {e}")

        if self.dynamic_sample:
            sample_rate = random.randint(1, self.sample_rate)
        else:
            sample_rate = self.sample_rate


        required_frames = self.sequence_length * sample_rate
        if total_frames < self.sequence_length:
            raise RuntimeError(f"Video {video_path} has only {total_frames} frames, But {self.sequence_length} are required.")

        if total_frames < required_frames:
            sample_rate = 1
            required_frames = self.sequence_length

        start_frame_ind = random.randint(0, max(0, total_frames - required_frames))
        end_frame_ind = min(start_frame_ind + required_frames, total_frames)
        frame_indice = np.linspace(
            start_frame_ind, end_frame_ind-1, self.sequence_length, dtype=int
        )
        video_data = decord_vr.get_batch(frame_indice).asnumpy()
        video_data = torch.from_numpy(video_data).float()
        video_data = video_data.permute(0, 3, 1, 2)

        video = self.norm(video_data).permute(1, 0, 2, 3)
        return {"video": video}
    
        

        





class VideoData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def _dataset(self, train):

        datasets = []
        for dataset_path, train_list, val_list in zip(self.args.data_path, self.args.train_dataset, self.args.val_dataset):

            dataset = MultiSizeVideoDataset(
                data_list=train_list if train else val_list, sequence_length=self.args.sequence_length,
                data_folder=dataset_path,
                train=train,
                sample_rate=self.args.sample_rate,
                dynamic_sample=self.args.dynamic_sample
                )
            datasets.append(dataset)

        return datasets


    def _dataloader(self,
                    train,
                    steps=0,
                    batch_size=None):

        dataset = self._dataset(train)
        if isinstance(self.args.batch_size, int):
            self.args.batch_size = [self.args.batch_size]

        self.batch_size = self.args.batch_size if batch_size is None else batch_size
        assert len(dataset) == len(self.args.batch_size)

        dataloaders = []
        for dset, d_batch_size in zip(dataset, self.batch_size):
            if dist.is_initialized():
                sampler = LengthGroupedSampler(batch_size=d_batch_size,
                                               world_size=dist.get_world_size(),
                                               gradient_accumulation_size=1,
                                               initial_global_step=steps if train else 0,
                                               lengths=dset.lengths,
                                               group_data=True,
                                               rank=dist.get_rank())
            else:
                sampler = None

            dataloader = data.DataLoader(dataset=dset,
                                         batch_size=d_batch_size,
                                         num_workers=self.args.num_workers if train else 0,
                                         pin_memory=False,
                                         sampler=sampler)
            dataloaders.append(dataloader)
        return dataloaders

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)[0]


