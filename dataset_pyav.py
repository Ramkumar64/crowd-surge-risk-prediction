import pandas as pd
import torch
from torch.utils.data import Dataset
import av, os
import numpy as np
import torchvision.transforms as T


CLASS_NAMES = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

def make_transform(image_size=224, train=True):
    if train:
        return T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(int(image_size*1.12)),
            T.RandomResizedCrop(image_size, scale=(0.65,1.0)),
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

class WindowVideoDatasetPyAV(Dataset):
    def __init__(self, csv_file, video_root, frames=32, train=True, image_size=224):
        self.df = pd.read_csv(csv_file)
        self.video_root = video_root.rstrip("/")
        self.frames = frames
        self.train = train
        self.transform = make_transform(image_size=image_size, train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.video_root, row.video)
        start = int(row.win_start); end = int(row.win_end)
        container = av.open(path)
        stream = container.streams.video[0]
        # decode sequentially, collect needed frames
        frames=[]
        for i, frame in enumerate(container.decode(stream)):
            if i > end: break
            if i >= start:
                img = frame.to_ndarray(format="rgb24")
                frames.append(img)
        container.close()
        if len(frames)==0:
            frames = [np.zeros((224,224,3), dtype=np.uint8)] * self.frames
        if len(frames) < self.frames:
            frames.extend([frames[-1]]*(self.frames - len(frames)))
        elif len(frames) > self.frames:
            frames = frames[:self.frames]
        tensors=[]
        for f in frames:
            t = torch.from_numpy(f).permute(2,0,1).float()/255.
            t = self.transform(t)
            tensors.append(t)
        clip = torch.stack(tensors, dim=0)
        label = CLASS_TO_IDX[row.label.replace(" ","_")]
        return clip, label