#!/usr/bin/env python3
"""
Single-file script: classify crowd behavior on a video using a trained temporal GRU model
(compatible with the feature+motion pipeline you trained).

Usage example:
python run_infer.py \
  --video /path/to/video.mp4 \
  --checkpoint best_temporal.pt \
  --out_csv results.csv \
  --win_size 32 --stride 16 --T 32 --device mps

Outputs:
- CSV with per-window predictions + probabilities + risk_score + smoothed_risk
- Prints top summary
"""

import os, cv2, argparse, torch, torchvision, torch.nn as nn
import numpy as np
import pandas as pd

CLASS_NAMES = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]
SEVERITY = {"Normal":0,"Light_Panic":1,"Fight":2,"Violent_Group":3,"Stampede":4}

# ---------------- Model (must match training) ----------------
class GRUClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=256, layers=1, bidir=True):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, num_classes)
        )
    def forward(self, x):          # x: (B,T,D)
        out, _ = self.gru(x)
        rep = out[:, -1, :]        # last time step
        return self.head(rep)

# --------------- Embedding Extraction (ResNet18 + motion) ---------------
def load_backbone(device):
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.eval().to(device)
    return m

def extract_embeddings(video_path, device, resize=128, every=1):
    back = load_backbone(device)
    cap = cv2.VideoCapture(video_path)
    feats=[]; motions=[]; prev_gray=None
    tfm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((resize, resize)),
        torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    idx=0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % every != 0:
                idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t = tfm(rgb).unsqueeze(0).to(device)
            f = back(t).squeeze(0).cpu()          # (512,)
            mot = 0.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean())
            prev_gray = gray
            feats.append(f)
            motions.append(mot)
            idx += 1
    cap.release()
    if not feats:
        emb = torch.zeros(1, 513)
    else:
        emb = torch.stack(feats, 0)                 # (N,512)
        mot = torch.tensor(motions).unsqueeze(1)    # (N,1)
        emb = torch.cat([emb, mot], 1)              # (N,513)
    return emb  # (N,513)

# --------------- Windowing + Sampling ---------------
def make_windows(num_frames, win_size, stride):
    windows=[]
    pos=0
    while pos + win_size <= num_frames:
        windows.append((pos, pos+win_size-1))
        pos += stride
    # tail
    if num_frames >= win_size and num_frames - (pos - stride) > win_size//2 and (num_frames - win_size) > 0:
        tail_start = num_frames - win_size
        if not windows or windows[-1][0] != tail_start:
            windows.append((tail_start, tail_start+win_size-1))
    return windows

def sample_uniform(feats_slice, T):
    L = feats_slice.shape[0]
    if L >= T:
        idxs = np.linspace(0, L-1, T, dtype=int)
        return feats_slice[idxs]
    pad = feats_slice[-1:].repeat(T-L, 1)
    return torch.cat([feats_slice, pad], 0)

# --------------- Risk & Smoothing ---------------
def ema(series, alpha=0.3):
    out=[]; prev=None
    for x in series:
        prev = x if prev is None else alpha*x + (1-alpha)*prev
        out.append(prev)
    return out

# --------------- Main ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_csv", default="results.csv")
    ap.add_argument("--win_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--every", type=int, default=1, help="Use every Nth frame when embedding")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ema_alpha", type=float, default=0.3)
    ap.add_argument("--alert_threshold", type=float, default=2.5)
    args = ap.parse_args()

    device = args.device
    if device=="cuda" and not torch.cuda.is_available(): device="cpu"
    if device=="mps" and not torch.backends.mps.is_available(): device="cpu"

    print(f"[INFO] Extracting embeddings: {args.video}")
    feats = extract_embeddings(args.video, device, every=args.every)  # (N,513)
    N, D = feats.shape
    print(f"[INFO] Frames embedded: {N}")

    print("[INFO] Building model + loading checkpoint")
    model = GRUClassifier(in_dim=D, num_classes=len(CLASS_NAMES))
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    windows = make_windows(N, args.win_size, args.stride)
    print(f"[INFO] Windows: {len(windows)}")

    softmax = nn.Softmax(dim=1)
    rows=[]
    with torch.no_grad():
        for (s,e) in windows:
            slice_feats = feats[s:e+1]          # (win_size, D) or shorter
            clip = sample_uniform(slice_feats, args.T).unsqueeze(0).to(device)  # (1,T,D)
            logits = model(clip)
            probs = softmax(logits).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            risk = float(sum(SEVERITY[c]*probs[i] for i,c in enumerate(CLASS_NAMES)))
            rows.append({
                "start_frame": s,
                "end_frame": e,
                "pred": pred_label,
                **{f"p_{c}": float(probs[i]) for i,c in enumerate(CLASS_NAMES)},
                "risk_score": risk
            })

    df = pd.DataFrame(rows)
    df["risk_ema"] = ema(df["risk_score"].tolist(), alpha=args.ema_alpha)
    df["alert"] = (df["risk_ema"] >= args.alert_threshold).astype(int)
    df.to_csv(args.out_csv, index=False)

    # Summary
    counts = df["pred"].value_counts()
    print("\nPrediction counts:")
    for c in CLASS_NAMES:
        print(f"{c:15s} {counts.get(c,0)}")
    print(f"\nSaved: {args.out_csv}")
    print(df.head())

if __name__ == "__main__":
    main()