#!/usr/bin/env python3
"""
Overlay event classification + risk score directly onto a video.

Pipeline (single pass):
1. Extract per-frame embeddings (ResNet18 + motion magnitude) for all frames.
2. Form sliding windows (win_size, stride).
3. For each window: uniformly sample to T frames, run temporal GRU model (trained earlier on 513-dim inputs).
4. Assign window prediction + risk score to all frames inside that window (if overlapping windows -> last or max-risk).
5. Optional EMA smoothing on per-frame risk.
6. Render annotated video with class label, probabilities (top-1 only by default), raw & smoothed risk bars.

Requirements:
- OpenCV, torch, torchvision, numpy, pandas (pandas only if exporting CSV).
- Your trained checkpoint (e.g. best_temporal.pt) produced by the earlier GRU pipeline (feature dim 513).

Usage example:
python visualize_video.py \
  --video '/Users/ithikash/Downloads/Crowd-Analysis-main/HAJJv2.Dataset/Training/Videos/Charlie Kirk shooting_ Video captures panic at Utah campus as gunshot heard.mp4' \
  --checkpoint best_temporal.pt \
  --out_video annotated.mp4 \
  --win_size 32 --stride 16 --T 32 \
  --every 1 --device mps \
  --ema_alpha 0.3 --fps_override 0

To also save per-window CSV:
  add: --out_csv predictions.csv
"""

import os, cv2, argparse, torch, torchvision, torch.nn as nn
import numpy as np
import pandas as pd

# ---------------- Configuration ----------------
CLASS_NAMES = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]
SEVERITY    = {"Normal":0,"Light_Panic":1,"Fight":2,"Violent_Group":3,"Stampede":4}

# Color (B,G,R) per class
CLASS_COLOR = {
    "Normal":        (40,180,40),
    "Light_Panic":   (0,215,255),
    "Fight":         (30,30,255),
    "Violent_Group": (0,70,255),
    "Stampede":      (0,0,0)  # outlined with yellow fill later
}

# ---------------- Model (must match training) ----------------
class GRUClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=256, layers=1, bidir=True):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers,
                          batch_first=True, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, num_classes)
        )
    def forward(self, x):  # x: (B,T,D)
        out,_ = self.gru(x)
        rep = out[:, -1, :]
        return self.head(rep)

# ---------------- Embedding extraction ----------------
def load_backbone(device):
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.eval().to(device)
    return m

def extract_embeddings(video_path, device, resize=128, every=1):
    cap = cv2.VideoCapture(video_path)
    feats=[]; motions=[]; prev_gray=None
    tfm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((resize, resize)),
        torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    backbone = load_backbone(device)
    idx=0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % every != 0:
                idx += 1
                continue
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t = tfm(rgb).unsqueeze(0).to(device)
            f = backbone(t).squeeze(0).cpu()
            mot = 0.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean())
            prev_gray = gray
            feats.append(f)
            motions.append(mot)
            idx += 1
    cap.release()
    if not feats:
        arr = torch.zeros(1, 513)
    else:
        fstack = torch.stack(feats, 0)
        mstack = torch.tensor(motions).unsqueeze(1)
        arr = torch.cat([fstack, mstack], 1)
    return arr  # (N,513)

# ---------------- Window helpers ----------------
def build_windows(N, win_size, stride):
    ws=[]
    pos=0
    while pos + win_size <= N:
        ws.append((pos, pos+win_size-1))
        pos += stride
    # tail window if leftover
    if N >= win_size and N - (pos - stride) > win_size//2 and (N - win_size) > 0:
        tstart = N - win_size
        if not ws or ws[-1][0] != tstart:
            ws.append((tstart, tstart+win_size-1))
    return ws

def sample_uniform(fr_feats, T):
    L = fr_feats.shape[0]
    if L >= T:
        idxs = np.linspace(0, L-1, T, dtype=int)
        return fr_feats[idxs]
    pad = fr_feats[-1:].repeat(T-L, 1)
    return torch.cat([fr_feats, pad], 0)

# ---------------- EMA ----------------
def ema(series, alpha):
    out=[]; prev=None
    for x in series:
        prev = x if prev is None else alpha*x + (1-alpha)*prev
        out.append(prev)
    return out

# ---------------- Visualization helpers ----------------
def put_label(frame, text, color, y=28):
    cv2.rectangle(frame, (5,5), (5+len(text)*9+10, 5+26), color, -1)
    cv2.putText(frame, text, (12,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def draw_risk_bar(frame, risk, risk_ema, max_score=4.0):
    h,w = frame.shape[:2]
    bar_w = int((risk / max_score) * (w-20))
    cv2.rectangle(frame, (10, h-30), (10+bar_w, h-20), (0,0,255), -1)
    ema_w = int((risk_ema / max_score) * (w-20))
    cv2.rectangle(frame, (10, h-15), (10+ema_w, h-6), (0,255,255), -1)
    cv2.putText(frame, f"risk={risk:.2f} ema={risk_ema:.2f}", (10, h-35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

def class_color(label):
    c = CLASS_COLOR.get(label, (128,128,128))
    if label == "Stampede":
        # highlight
        return (0,255,255)  # bright yellow
    return c

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_video", default="annotated.mp4")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--win_size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--every", type=int, default=1, help="Use every Nth frame for embeddings & inference")
    ap.add_argument("--resize_embed", type=int, default=128)
    ap.add_argument("--device", default="mps", choices=["cpu","cuda","mps"])
    ap.add_argument("--ema_alpha", type=float, default=0.3)
    ap.add_argument("--alert_threshold", type=float, default=2.5)
    ap.add_argument("--fps_override", type=float, default=0.0, help="If >0, force output FPS")
    ap.add_argument("--codec", default="mp4v")
    args = ap.parse_args()

    device = args.device
    if device=="cuda" and not torch.cuda.is_available(): device="cpu"
    if device=="mps" and not torch.backends.mps.is_available(): device="cpu"
    print(f"[INFO] Device: {device}")

    # Step 1: Extract embeddings
    feats = extract_embeddings(args.video, device, resize=args.resize_embed, every=args.every)
    N, D = feats.shape
    print(f"[INFO] Embeddings shape: {feats.shape}")

    # Step 2: Windows
    windows = build_windows(N, args.win_size, args.stride)
    print(f"[INFO] Windows: {len(windows)}")

    # Step 3: Load model
    model = GRUClassifier(D, len(CLASS_NAMES))
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    softmax = nn.Softmax(dim=1)

    # Step 4: Window predictions
    win_preds = []
    with torch.no_grad():
        for (s,e) in windows:
            slice_feat = feats[s:e+1]
            clip = sample_uniform(slice_feat, args.T).unsqueeze(0).to(device)  # (1,T,D)
            logits = model(clip)
            probs = softmax(logits).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            risk = float(sum(SEVERITY[c]*probs[i] for i,c in enumerate(CLASS_NAMES)))
            win_preds.append({
                "start": s,
                "end": e,
                "pred": pred_label,
                "risk": risk,
                "probs": probs
            })

    # Step 5: Assign frame-level (simple: use last window that covers frame)
    frame_pred = ["Normal"] * N
    frame_risk = [0.0] * N
    for wp in win_preds:
        for f in range(wp["start"], wp["end"]+1):
            if 0 <= f < N:
                frame_pred[f] = wp["pred"]
                frame_risk[f] = wp["risk"]

    # Step 6: Smooth risk
    risk_ema = ema(frame_risk, alpha=args.ema_alpha)

    # Step 7: Prepare video writer
    cap_in = cv2.VideoCapture(args.video)
    in_fps = cap_in.get(cv2.CAP_PROP_FPS)
    fps_out = args.fps_override if args.fps_override > 0 else (in_fps if in_fps>0 else 25)
    w  = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out = cv2.VideoWriter(args.out_video, fourcc, fps_out, (w,h))
    if not out.isOpened():
        raise RuntimeError("Cannot open VideoWriter")

    # Map embedding frames to original frames (since we may skip)
    # If every=1 -> direct alignment. If every>1 => embedding frame i corresponds to original frame i*every
    frame_map = {}  # original_frame_index -> embedding_frame_index
    emb_idx=0
    for orig in range(int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))):
        if orig % args.every == 0 and emb_idx < N:
            frame_map[orig] = emb_idx
            emb_idx += 1

    print(f"[INFO] Rendering annotated video -> {args.out_video}")
    frame_id = 0
    while True:
        ret, frame = cap_in.read()
        if not ret: break
        if frame_id in frame_map:
            ei = frame_map[frame_id]
            lab = frame_pred[ei]
            risk = frame_risk[ei]
            rem = risk_ema[ei]
            clr = class_color(lab)
            put_label(frame, f"{lab}", clr)
            draw_risk_bar(frame, risk, rem)
            # Alert overlay
            if rem >= args.alert_threshold and lab != "Normal":
                cv2.putText(frame, "ALERT", (w-160, 40), cv2.FONT_HERSHEY_DUPLEX,
                            1.2, (0,0,255), 3, cv2.LINE_AA)
        out.write(frame)
        frame_id += 1

    cap_in.release()
    out.release()
    print("[INFO] Saved annotated video:", args.out_video)

    # Optional CSV
    if args.out_csv:
        rows=[]
        for wp in win_preds:
            row = {
                "start_frame": wp["start"] * args.every,
                "end_frame":   wp["end"] * args.every,
                "pred": wp["pred"],
                "risk": wp["risk"]
            }
            for i,c in enumerate(CLASS_NAMES):
                row[f"p_{c}"] = float(wp["probs"][i])
            rows.append(row)
        df = pd.DataFrame(rows)
        df["risk_ema_window"] = ema(df["risk"].tolist(), alpha=args.ema_alpha)
        df.to_csv(args.out_csv, index=False)
        print("[INFO] Saved window CSV:", args.out_csv)

if __name__ == "__main__":
    main()