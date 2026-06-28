import os, argparse, cv2, torch, torchvision
import torch.nn as nn
from tqdm import tqdm

def get_model(device):
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.eval().to(device)
    return m, 512

def process_video(path, model, feat_dim, device, resize=128, every=1):
    cap = cv2.VideoCapture(path)
    frames_feats = []
    prev_gray = None
    motion_list = []
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
            feat = model(t)  # (1,512)
            if prev_gray is None:
                motion = 0.0
            else:
                # simple motion magnitude
                diff = cv2.absdiff(gray, prev_gray)
                motion = float(diff.mean())
            prev_gray = gray
            # append motion as extra dimension
            feat_with_motion = torch.cat([feat.squeeze(0), torch.tensor([motion], device=device)], dim=0)
            frames_feats.append(feat_with_motion.cpu())
            idx += 1
    cap.release()
    if not frames_feats:
        return torch.zeros(1, feat_dim+1), 0
    feats = torch.stack(frames_feats, dim=0)  # (N, 513)
    return feats, feats.shape[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", default="embeddings")
    ap.add_argument("--every", type=int, default=1, help="Take every Nth frame")
    ap.add_argument("--device", default="mps", help="cuda|mps|cpu")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    os.makedirs(args.out_dir, exist_ok=True)
    model, feat_dim = get_model(device)

    vids = [f for f in os.listdir(args.video_dir) if f.lower().endswith(".mp4")]
    for v in vids:
        stem = os.path.splitext(v)[0]
        out_path = os.path.join(args.out_dir, stem + ".pt")
        if os.path.exists(out_path):
            print("Skip existing", v)
            continue
        path = os.path.join(args.video_dir, v)
        feats, count = process_video(path, model, feat_dim, device, every=args.every)
        torch.save({"features": feats, "frame_count": count}, out_path)
        print(f"Saved {out_path}, frames={count}")

if __name__ == "__main__":
    main()