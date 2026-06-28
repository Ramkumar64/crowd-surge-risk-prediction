import pandas as pd, cv2, os, torch, numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

CLASS_NAMES = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASS_NAMES)}

def build_transform(image_size=224, train=True):
    if train:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(int(image_size*1.12)),
            T.RandomResizedCrop(image_size, scale=(0.6,1.0)),
            T.RandomHorizontalFlip(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(int(image_size*1.12)),
            T.CenterCrop(image_size),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

class TimeWindowDataset(Dataset):
    def __init__(self, csv_file, video_root, frames_per_clip=16,
                 image_size=224, train=True, seed=42):
        self.df = pd.read_csv(csv_file)
        self.video_root = video_root.rstrip("/")
        self.frames_per_clip = frames_per_clip
        self.transform = build_transform(image_size, train=train)
        self.train = train
        random.seed(seed)

    def __len__(self):
        return len(self.df)

    def _open_video_range(self, path, start_f, end_f):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_f = max(0, min(start_f, total-1))
        end_f   = max(0, min(end_f, total-1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frames=[]
        cur = start_f
        while cur <= end_f:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            cur += 1
        cap.release()
        return frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_path = os.path.join(self.video_root, row.video)
        start = int(row.win_start)
        end   = int(row.win_end)
        frames = self._open_video_range(vid_path, start, end)

        if len(frames) == 0:
            frames = [np.zeros((224,224,3), dtype=np.uint8)]

        # Uniform sampling to frames_per_clip
        T_total = len(frames)
        if T_total >= self.frames_per_clip:
            # indices spaced linearly
            idxs = np.linspace(0, T_total-1, self.frames_per_clip, dtype=int)
            sampled = [frames[i] for i in idxs]
        else:
            # pad by repeating last
            sampled = frames + [frames[-1]]*(self.frames_per_clip - T_total)

        tensors=[]
        for f in sampled:
            t = torch.from_numpy(f).permute(2,0,1).float()/255.
            t = self.transform(t)
            tensors.append(t)
        clip = torch.stack(tensors, dim=0)     # (T,C,H,W)
        label = CLASS_TO_IDX[row.label.replace(" ","_")]
        return clip, label