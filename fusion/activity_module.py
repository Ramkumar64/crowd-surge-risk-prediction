"""
activity_module.py
Embeds frames (ResNet18 + motion) and provides rolling temporal activity
probabilities using a GRU head trained previously (513-dim input).

If no checkpoint is supplied, returns a neutral probability vector
[1,0,0,0,0] (i.e. Normal only).
"""

import cv2, torch, numpy as np, torchvision, torch.nn as nn
from collections import deque
from typing import Dict, List

ACTIVITY_CLASSES = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]
NUM_ACT = len(ACTIVITY_CLASSES)

class GRUClassifier(nn.Module):
    def __init__(self, in_dim=513, hidden=256, num_classes=NUM_ACT, bidir=True):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=bidir)
        out_dim = hidden*(2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, num_classes)
        )
    def forward(self, x):  # (B,T,F)
        out,_ = self.gru(x)
        rep = out[:,-1,:]
        return self.head(rep)

class ActivityModule:
    def __init__(self,
                 T=32,
                 stride=16,
                 device="cpu",
                 ckpt_path=None,
                 resize=128):
        self.T = T
        self.stride = stride
        self.device = device
        self.resize = resize
        self.buffer = deque(maxlen=T)
        self.prev_gray = None
        self.frame_counter = 0

        self.backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.backbone.eval().to(device)

        self.enabled = ckpt_path is not None and ckpt_path != "" and os_path_exists(ckpt_path)
        if self.enabled:
            self.classifier = GRUClassifier().to(device)
            state = torch.load(ckpt_path, map_location="cpu")
            self.classifier.load_state_dict(state)
            self.classifier.eval()
        else:
            self.classifier = None

        self.softmax = nn.Softmax(dim=1)
        self.last_probs = np.array([1.0]+[0.0]*(NUM_ACT-1), dtype=np.float32)

        self.tfm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((resize, resize)),
            torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def step(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mot = 0.0 if self.prev_gray is None else float(cv2.absdiff(gray, self.prev_gray).mean())
        self.prev_gray = gray

        with torch.no_grad():
            t = self.tfm(rgb).unsqueeze(0).to(self.device)
            feat = self.backbone(t).squeeze(0).cpu()  # (512,)
        feat513 = torch.cat([feat, torch.tensor([mot])], dim=0)  # (513,)
        self.buffer.append(feat513)
        self.frame_counter += 1

        if self.enabled and len(self.buffer) == self.T and (self.frame_counter % self.stride == 0):
            clip = torch.stack(list(self.buffer), dim=0).unsqueeze(0).to(self.device)  # (1,T,513)
            with torch.no_grad():
                logits = self.classifier(clip)
                probs = self.softmax(logits).cpu().numpy()[0]
            self.last_probs = probs.astype(np.float32)

        return self.last_probs

    @staticmethod
    def stats_from_probs(probs):
        entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        top2 = np.sort(probs)[-2:]
        gap = float(top2[-1] - (top2[-2] if len(top2)==2 else 0))
        abnormal = float(1.0 - probs[0])
        return {
            "act_entropy": entropy,
            "act_top_prob": float(np.max(probs)),
            "act_top2_gap": gap,
            "act_abnormal_prob": abnormal
        }

def os_path_exists(p):
    import os
    return os.path.isfile(p)

import numpy as np  # keep at bottom for faster import if missing earlier