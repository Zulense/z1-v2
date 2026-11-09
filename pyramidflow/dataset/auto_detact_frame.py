import torch 
from torch.utils.data import Dataset, DataLoader
import jsonlines
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import cv2
import random
from pathlib import Path


class VideoDataset(Dataset):

    def __init__(self,
                 anno_file,
                 resolution=256,
                 add_normalize=True):

        super().__init__()
        self.video_annos = []

        if not isinstance(anno_file, list):
            anno_file = [anno_file]

        print("The training video clip will use all available frames.")

        for anno_file_ in anno_file:
            print(f"Load annotation file from {anno_file_}")

            with jsonlines.open(anno_file_, 'r') as reader:
                for item in tqdm(reader):
                    self.video_annos.append(item)

        print(f"Totally Remained {len(self.video_annos)} videos")
        self.video_processor = VideoFrameProcessor(resolution=resolution,
                                                   add_normalize=add_normalize)

    def __len__(self):
        return len(self.video_annos)

    def __getitem__(self, index):
        import os # Ensure this is imported

        video_anno = self.video_annos[index]
        video_path = video_anno['video']

        # --- PATH CORRECTION ---
        # Update this logic to match the machine you are currently running on.
        # If testing locally on your desktop, ensure this points to your local /Data folder.
        if video_path.startswith("/Data/"):
            # Example for AIRAWAT cluster:
            # video_path = f"/nlsasfs/home/hfgenai/mansav{video_path}"
            
            # Example for local desktop (Update this to your actual local dataset path):
            video_path = f"/home/manish/Desktop/projects/z1-v2{video_path}" 
        
        # 1. Fail fast and loud if the path is entirely wrong
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Path Error: Could not find video at {video_path}")
        
        try:
            video_tensors, video_frames = self.video_processor(video_path)
            
            if video_tensors is None:
                raise ValueError(f"Video processor returned None for {video_path}")

            return {
                "video": video_tensors,
                "identifier": "video"
            }

        except Exception as e:
            print(f"Loading video Error with {e}")
            # 2. Only recursively try another video if it's an isolated corrupted file, 
            # not a missing file path issue.
            return self.__getitem__(random.randint(0, self.__len__() -1))

    


class VideoFrameProcessor:

    # load a video and transform.
    def __init__(self,
                 resolution=256,
                 add_normalize=True,
                 sample_fps=24):
        
        image_size = resolution

        transform_list = [
            transforms.Resize(image_size,
                              interpolation=InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.CenterCrop(image_size)
        ]

        if add_normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                       std=(0.5, 0.5, 0.5)))

        print(f"Transform List is {transform_list}")
        self.transform = transforms.Compose(transform_list)
        self.sample_fps = sample_fps

    def __call__(self, video_path):
        try:
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frames = []

            while True:
                flag, frame = video_capture.read()
                if not flag:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)

            video_capture.release()
            
            if len(frames) == 0:
                return None, None

            sample_fps = self.sample_fps
            interval = max(int(fps / sample_fps), 1)
            frames = frames[::interval]

            # Process and stack all available frames (no padding or slicing required)
            filtered_frames = torch.stack(frames).float() / 255
            filtered_frames = self.transform(filtered_frames)
            
            # Permute to match [C, T, H, W] format
            filtered_frames = filtered_frames.permute(1, 0, 2, 3)

            return filtered_frames, None

        except Exception as e:
            print(f"Load video: {video_path} Error, Exception {e}")
            return None, None


if __name__ == "__main__":

    training_dataset = VideoDataset(anno_file="/home/manish/Desktop/projects/z1-v2/annotation/class_5_math_white_board.jsonl"
                                    )

    print(training_dataset)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset=training_dataset,
                        batch_size=1)

    for data in loader:
        print(data["video"].shape)