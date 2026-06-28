import pandas as pd, cv2, os, torch, numpy as np
from torch.utils.data import Dataset

CLASS_NAMES = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASS_NAMES)}

class VideoMAEWindowDataset(Dataset):
    def __init__(self, csv_file, video_root, frames_per_clip=16,
                 train=True, seed=42):
        self.df = pd.read_csv(csv_file)
        self.video_root = video_root.rstrip("/")
        self.frames_per_clip = frames_per_clip
        self.train = train
        self.rng = np.random.RandomState(seed)

    def __len__(self): return len(self.df)

    def _read_range(self, path, start_f, end_f):
        cap = cv2.VideoCapture(path)
        frames=[]
        if not cap.isOpened():
            return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_f = max(0, min(start_f, total-1))
        end_f   = max(0, min(end_f, total-1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        cur = start_f
        while cur <= end_f:
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cur += 1
        cap.release()
        return frames

    def _uniform(self, frames):
        T = len(frames)
        if T >= self.frames_per_clip:
            idxs = np.linspace(0, T-1, self.frames_per_clip, dtype=int)
            return [frames[i] for i in idxs]
        return frames + [frames[-1]]*(self.frames_per_clip - T)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_path = os.path.join(self.video_root, row.video)
        start, end = int(row.win_start), int(row.win_end)
        raw = self._read_range(vid_path, start, end)
        if not raw:
            raw = [np.zeros((224,224,3), dtype=np.uint8)]
        frames = self._uniform(raw)
        label = CLASS_TO_IDX[row.label.replace(" ","_")]
        return frames, label