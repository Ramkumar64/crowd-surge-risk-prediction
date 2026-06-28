import argparse, time, os, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from focal_loss import FocalLoss  # Ensure focal_loss.py is in same folder

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CLASS_NAMES_FULL = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]
CLASS_TO_IDX_FULL = {c:i for i,c in enumerate(CLASS_NAMES_FULL)}

# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class FeatureWindowDataset(Dataset):
    """
    Loads pre-extracted per-frame embeddings (.pt files) and slices windows.

    Each embeddings file: { "features": Tensor (N, D), "frame_count": int }
    CSV columns: video,label,win_start,win_end,...
    """
    def __init__(self, csv_file, embeddings_dir, T=32, cache=True):
        self.df = pd.read_csv(csv_file)
        self.emb_dir = embeddings_dir
        self.T = T
        self.cache = cache
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def _load_video_feats(self, video_name):
        stem = os.path.splitext(video_name)[0]
        path = os.path.join(self.emb_dir, stem + ".pt")
        if self.cache and path in self._cache:
            return self._cache[path]
        data = torch.load(path, map_location="cpu")
        feats = data["features"]  # (N, D)
        if self.cache:
            self._cache[path] = feats
        return feats

    def _uniform_indices(self, length):
        if length >= self.T:
            # Uniform spacing
            idxs = np.linspace(0, length - 1, self.T, dtype=int)
        else:
            # Pad last frame
            base = list(range(length))
            idxs = base + [length - 1] * (self.T - length)
        return idxs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feats_all = self._load_video_feats(row.video)
        start = int(row.win_start)
        end = int(row.win_end)
        end = min(end, feats_all.shape[0] - 1)
        start = min(start, end)
        window_feats = feats_all[start:end+1]  # (L, D)
        idxs = self._uniform_indices(window_feats.shape[0])
        sampled = window_feats[idxs]  # (T, D)
        label_text = row.label.replace(" ", "_")
        label_idx = CLASS_TO_IDX_FULL[label_text]
        return sampled, label_idx

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
class MeanPoolClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: (B,T,D)
        x = x.mean(1)  # (B,D)
        x = self.norm(x)
        return self.fc(x)

class GRUClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=256, layers=1, bidir=True, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=bidir, dropout=0 if layers == 1 else dropout)
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, x):  # (B,T,D)
        out, _ = self.gru(x)
        rep = out[:, -1, :]  # last time step
        return self.head(rep)

class TemporalConvClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, channels=256, kernel=5, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, channels, kernel, padding=kernel//2)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=kernel//2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):  # (B,T,D)
        x = x.transpose(1, 2)  # (B,D,T)
        x = self.act(self.conv1(x))
        x = self.drop(self.act(self.conv2(x)))
        x = x.mean(-1)  # (B,C)
        return self.fc(x)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def build_model(kind, in_dim, num_classes):
    if kind == "mean":
        return MeanPoolClassifier(in_dim, num_classes)
    if kind == "gru":
        return GRUClassifier(in_dim, num_classes, hidden=256, bidir=True)
    if kind == "tconv":
        return TemporalConvClassifier(in_dim, num_classes)
    raise ValueError(f"Unknown model_type={kind}")

def compute_alpha(counts):
    counts = np.array(counts, dtype=float)
    inv = 1.0 / np.maximum(counts, 1)
    alpha = inv / inv.sum()
    return torch.tensor(alpha, dtype=torch.float32)

def make_sampler(labels):
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=len(np.unique(labels)))
    weights = 1.0 / np.maximum(counts, 1)
    sample_w = weights[labels]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(labels)*2, replacement=True)
    return sampler, counts

def collate(batch, label_map_fn):
    feats, labels = [], []
    for x, orig_idx in batch:
        mapped = label_map_fn(CLASS_NAMES_FULL[orig_idx])
        feats.append(x)
        labels.append(mapped)
    feats = torch.stack(feats, dim=0)  # (B,T,D)
    labels = torch.tensor(labels)
    return feats, labels

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="balanced_train.csv")
    ap.add_argument("--val_csv", default="balanced_val.csv")
    ap.add_argument("--embeddings_dir", default="embeddings")
    ap.add_argument("--model_type", choices=["mean","gru","tconv"], default="gru")
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--sampler", action="store_true")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--binary", action="store_true", help="Collapse to Normal vs NonNormal.")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available(): device = "cpu"
    if device == "mps"  and not torch.backends.mps.is_available(): device = "cpu"

    print(f"[CONFIG] device={device} model={args.model_type} T={args.T} binary={args.binary}")

    # Datasets
    train_ds = FeatureWindowDataset(args.train_csv, args.embeddings_dir, T=args.T, cache=True)
    val_ds   = FeatureWindowDataset(args.val_csv, args.embeddings_dir, T=args.T, cache=True)

    # Label mapping
    if args.binary:
        target_names = ["Normal","NonNormal"]
        def label_map_fn(lbl_text):
            return 0 if lbl_text == "Normal" else 1
    else:
        target_names = CLASS_NAMES_FULL
        def label_map_fn(lbl_text):
            return CLASS_TO_IDX_FULL[lbl_text]

    # Collect mapped train labels for counts / sampler
    mapped_train_labels = []
    for i in range(len(train_ds)):
        _, orig_idx = train_ds[i]
        mapped_train_labels.append(label_map_fn(CLASS_NAMES_FULL[orig_idx]))
    mapped_train_labels = np.array(mapped_train_labels)

    # Sampler or plain shuffle
    if args.sampler:
        sampler, counts = make_sampler(mapped_train_labels)
        print("Train counts (mapped):", counts)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, sampler=sampler, num_workers=0,
            collate_fn=lambda b: collate(b, label_map_fn)
        )
    else:
        counts = np.bincount(mapped_train_labels, minlength=len(target_names))
        print("Train counts (mapped):", counts)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, num_workers=0,
            collate_fn=lambda b: collate(b, label_map_fn)
        )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate(b, label_map_fn)
    )

    # Feature dimension
    sample_feat, _ = train_ds[0]
    feat_dim = sample_feat.shape[-1]

    model = build_model(args.model_type, feat_dim, len(target_names)).to(device)

    # Alpha / loss
    alpha = compute_alpha(counts).to(device)
    if args.use_focal:
        criterion = FocalLoss(alpha=alpha, gamma=args.gamma)
        print("Using Focal Loss alpha:", alpha.cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha)
        print("Using Weighted CE alpha:", alpha.cpu().numpy())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # Train
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += y.size(0)

        train_loss = total_loss / max(1, total_samples)
        train_acc = total_correct / max(1, total_samples)

        # Validate
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                val_preds.append(logits.argmax(1).cpu())
                val_true.append(y.cpu())
        val_preds = torch.cat(val_preds).numpy()
        val_true  = torch.cat(val_true).numpy()

        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        report = classification_report(
            val_true, val_preds,
            labels=list(range(len(target_names))),
            target_names=target_names,
            digits=3,
            zero_division=0
        )
        macro_f1 = f1_score(
            val_true, val_preds,
            labels=list(range(len(target_names))),
            average="macro",
            zero_division=0
        )
        cm = confusion_matrix(val_true, val_preds, labels=list(range(len(target_names))))

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch} | {elapsed:.1f}s | TrainLoss {train_loss:.4f} "
              f"Acc {train_acc:.3f} MacroF1 {macro_f1:.3f}")
        print(report)
        print("Confusion Matrix:")
        for r in cm: print(" ", r)

        # Check improvement
        if macro_f1 > best_f1 + 1e-4:
            best_f1 = macro_f1
            patience_ctr = 0
            torch.save(model.state_dict(), "best_temporal.pt")
            print("[INFO] Saved best_temporal.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print("[INFO] Early stopping.")
                break

        scheduler.step()
        torch.save(model.state_dict(), f"temporal_epoch{epoch}.pt")

if __name__ == "__main__":
    main()