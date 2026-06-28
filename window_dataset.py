import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2, os, numpy as np
import torchvision.transforms as T

CLASS_NAMES = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def _make_transform(image_size=224, train=True):
    if train:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(int(image_size*1.12)),
            T.RandomResizedCrop(image_size, scale=(0.6,1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2,0.2,0.2,0.1),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(int(image_size*1.12)),
            T.CenterCrop(image_size),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

class WindowVideoDatasetCV2Subsample(Dataset):
    """
    Loads temporal window (win_start, win_end). Applies temporal_step skipping,
    pads/truncates to 'frames', returns (T,C,H,W), label_idx.
    """
    def __init__(self,
                 csv_file,
                 video_root,
                 frames=32,
                 train=True,
                 image_size=224,
                 temporal_step=1):
        self.df = pd.read_csv(csv_file)
        self.video_root = video_root.rstrip("/")
        self.frames = frames
        self.train = train
        self.image_size = image_size
        self.temporal_step = max(1, temporal_step)
        self.transform = _make_transform(image_size=image_size, train=train)

    def __len__(self):
        return len(self.df)

    def _read_window(self, path, start, end):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        start = max(0, min(start, total - 1))
        end   = max(0, min(end,   total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        out = []
        cur = start
        while cur <= end:
            ret, frame = cap.read()
            if not ret:
                break
            if ((cur - start) % self.temporal_step) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.append(frame)
            cur += 1
        cap.release()
        return out

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.video_root, row.video)
        start = int(row.win_start)
        end   = int(row.win_end)

        frames_seq = self._read_window(video_path, start, end)

        if len(frames_seq) == 0:
            frames_seq = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)]

        # Pad / truncate
        if len(frames_seq) < self.frames:
            frames_seq.extend([frames_seq[-1]] * (self.frames - len(frames_seq)))
        elif len(frames_seq) > self.frames:
            frames_seq = frames_seq[:self.frames]

        tensors = []
        for f in frames_seq:
            t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            t = self.transform(t)
            tensors.append(t)
        clip = torch.stack(tensors, dim=0)  # (T,C,H,W)

        label = CLASS_TO_IDX[row.label.replace(" ", "_")]
        return clip, label