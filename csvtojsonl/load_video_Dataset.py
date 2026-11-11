import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import decord
import numpy as np

class OpenVidDataset(Dataset):
    def __init__(self, jsonl_path, num_frames=16, resize=(256, 256)):
        self.num_frames = num_frames
        self.samples = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
                    
        self.transform = T.Compose([
            T.Resize(resize, antialias=True),
            T.ToDtype(torch.float32, scale=True), # uint8 [0, 255] -> float32 [0.0, 1.0]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        video_path = item["video"]
        
        # Load video with decord
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)
        
        # Uniformly sample `self.num_frames` indices across the video
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        # Decode only the sampled frames -> Shape: (T, H, W, C)
        frames = vr.get_batch(indices).asnumpy()
        
        # Convert to tensor and change layout to (T, C, H, W)
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)
        video = self.transform(video)
        
        return {
            "video": video,  # Shape: (num_frames, 3, H, W)
            "caption": item["caption"],
            "aesthetic_score": torch.tensor(item["aesthetic_score"], dtype=torch.float32),
            "motion_score": torch.tensor(item["motion_score"], dtype=torch.float32),
            "temporal_consistency_score": torch.tensor(item["temporal_consistency_score"], dtype=torch.float32),
            "camera_motion": item["camera_motion"],
            "fps": torch.tensor(item["fps"], dtype=torch.float32),
            "seconds": torch.tensor(item["seconds"], dtype=torch.float32),
        }

if __name__ == "__main__":
    jsonl_file = "/home/manish/Desktop/projects/z1-v2/demo_OpenVidHD.jsonl"  # Update with your path
    
    dataset = OpenVidDataset(jsonl_path=jsonl_file, num_frames=16, resize=(256, 256))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    
    for batch in dataloader:
        print("Video batch shape:", batch["video"].shape)  # [2, 16, 3, 256, 256]
        print("Captions:", batch["caption"])
        break