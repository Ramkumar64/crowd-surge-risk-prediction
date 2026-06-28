import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import torchvision.transforms as T
import os


CLASS_NAMES = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def make_transform(image_size=224, train=True):
    if train:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(int(image_size*1.1)),
            T.RandomResizedCrop(image_size, scale=(0.7,1.0)),
            T.RandomHorizontalFlip(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(int(image_size*1.1)),
            T.CenterCrop(image_size),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

class WindowVideoDatasetTorchVision(Dataset):
    def __init__(self, csv_file, video_root, frames=32, train=True, fps_assume=30):
        self.df = pd.read_csv(csv_file)
        self.video_root = video_root.rstrip("/")
        self.frames = frames
        self.train = train
        self.transform = make_transform(train=train)
        self.fps_assume = fps_assume  # only used if we cannot read fps
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.video_root, row.video)
        start_f = int(row.win_start); end_f = int(row.win_end)
        # Convert frame indices to seconds (approx)
        # We don't know exact fps reliably here -> assume constant or store fps earlier.
        start_sec = start_f / self.fps_assume
        end_sec   = (end_f + 1) / self.fps_assume
        # read_video expects times (in seconds) when start_pts/end_pts not in pts units
        video, _, info = read_video(path, start_pts=start_sec, end_pts=end_sec)
        # video shape: (T, H, W, C)
        if video.shape[0] == 0:
            video = torch.zeros((self.frames, 224, 224, 3), dtype=torch.uint8)
        # Pad/trim
        if video.shape[0] < self.frames:
            last = video[-1:].clone()
            pad = last.repeat(self.frames - video.shape[0],1,1,1)
            video = torch.cat([video, pad], dim=0)
        elif video.shape[0] > self.frames:
            video = video[:self.frames]
        # Convert each frame
        frames=[]
        for i in range(video.shape[0]):
            t = video[i].permute(2,0,1).float()/255.
            t = self.transform(t)
            frames.append(t)
        clip = torch.stack(frames, dim=0)  # (T,C,H,W)
        label = CLASS_TO_IDX[row.label.replace(" ","_")]
        return clip, label