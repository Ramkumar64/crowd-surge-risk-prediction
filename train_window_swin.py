import argparse, time, torch, torch.nn as nn, timm, os
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from collections import Counter
from window_dataset import (
    WindowVideoDatasetCV2Subsample,
    CLASS_NAMES
)

# ---------------- Model ----------------
class TemporalSwin(nn.Module):
    def __init__(self,
                 backbone="swin_tiny_patch4_window7_224",
                 num_classes=5,
                 pretrained=True,
                 freeze_backbone=False,
                 img_size=224):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):  # x: (B,T,C,H,W)
        if x.dim() != 5:
            raise ValueError(f"Expected (B,T,C,H,W) got {x.shape}")
        B, T, C, H, W = x.shape
        x = x.contiguous().reshape(B * T, C, H, W)
        feat = self.backbone(x)          # (B*T,F)
        feat = feat.reshape(B, T, -1).mean(1)
        return self.fc(feat)

# --------------- Collate ----------------
def collate_fn(batch):
    clips, labels = zip(*batch)
    clips = torch.stack(clips, dim=0)  # (B,T,C,H,W)
    labels = torch.tensor(labels)
    return clips, labels

# -------- Utility: Oversample Train --------
def oversample_indices(labels, strategy="max"):
    """Return an index list with oversampled minority classes."""
    counts = Counter(labels)
    if strategy == "max":
        target = max(counts.values())
    else:
        return list(range(len(labels)))
    per_class_indices = {c: [] for c in counts}
    for i, lbl in enumerate(labels):
        per_class_indices[lbl].append(i)
    new_indices = []
    for c, idxs in per_class_indices.items():
        if len(idxs) == 0:
            continue
        reps = target // len(idxs)
        rem = target % len(idxs)
        new_indices.extend(idxs * reps + list(np.random.choice(idxs, rem, replace=True)))
    np.random.shuffle(new_indices)
    return new_indices

# -------- Utility: Auto-balance validation --------
def auto_balance_validation(train_csv, val_csv, out_train_csv, out_val_csv):
    tdf = pd.read_csv(train_csv)
    vdf = pd.read_csv(val_csv)
    present = set(vdf.label.unique())
    needed = set(CLASS_NAMES) - present
    if not needed:
        return False  # no change
    # Try to move one instance per missing class
    moved_rows = []
    for cls in needed:
        candidates = tdf[tdf.label == cls]
        if len(candidates) == 0:
            continue
        row = candidates.sample(1, random_state=42)
        moved_rows.append(row)
        tdf = tdf.drop(row.index)
        vdf = pd.concat([vdf, row], ignore_index=True)
    # Save only if something changed
    if moved_rows:
        tdf.to_csv(out_train_csv, index=False)
        vdf.to_csv(out_val_csv, index=False)
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="window_train.csv")
    ap.add_argument("--val_csv",   default="window_val.csv")
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--sample_frames", type=int, default=8)
    ap.add_argument("--temporal_step", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone", default="swin_tiny_patch4_window7_224")
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--unfreeze_epoch", type=int, default=3)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--oversample_train", action="store_true",
                    help="Oversample minority classes in training set.")
    ap.add_argument("--auto_balance_val", action="store_true",
                    help="Move one sample per missing class from train to val if absent.")
    args = ap.parse_args()

    if args.sample_frames > args.frames:
        raise ValueError("--sample_frames cannot exceed --frames")

    # Possibly auto-balance validation
    train_csv_used = args.train_csv
    val_csv_used = args.val_csv
    if args.auto_balance_val:
        modified = auto_balance_validation(
            args.train_csv, args.val_csv,
            "_tmp_train_bal.csv", "_tmp_val_bal.csv"
        )
        if modified:
            print("[INFO] Validation set auto-balanced to include missing classes.")
            train_csv_used = "_tmp_train_bal.csv"
            val_csv_used   = "_tmp_val_bal.csv"

    train_df = pd.read_csv(train_csv_used)
    val_df   = pd.read_csv(val_csv_used)

    print("[INFO] Train class counts BEFORE oversample:")
    print(train_df.label.value_counts())
    print("[INFO] Val class counts:")
    print(val_df.label.value_counts())

    # Device
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    use_amp = (args.amp and device == "cuda")

    print(f"[CONFIG] device={device} backbone={args.backbone} frames={args.frames} "
          f"sample_frames={args.sample_frames} temporal_step={args.temporal_step} img_size={args.img_size}")

    # Datasets
    train_ds_full = WindowVideoDatasetCV2Subsample(
        train_csv_used, args.video_root,
        frames=args.frames, train=True,
        image_size=args.img_size, temporal_step=args.temporal_step
    )
    val_ds = WindowVideoDatasetCV2Subsample(
        val_csv_used, args.video_root,
        frames=args.frames, train=False,
        image_size=args.img_size, temporal_step=args.temporal_step
    )

    # Oversampling (index-based) if requested
    if args.oversample_train:
        labels_list = train_df.label.map(lambda x: CLASS_NAMES.index(x)).tolist()
        new_indices = oversample_indices(labels_list, strategy="max")
        # Wrap a Subset-like dataset
        class OversampledWrapper(torch.utils.data.Dataset):
            def __init__(self, base, indices):
                self.base = base
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                return self.base[self.indices[i]]
        train_ds = OversampledWrapper(train_ds_full, new_indices)
        print(f"[INFO] Oversampling active. Original train windows={len(train_ds_full)}, after oversample={len(train_ds)}")
    else:
        train_ds = train_ds_full

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=(device == "cuda"))

    # Model
    model = TemporalSwin(
        backbone=args.backbone,
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        img_size=args.img_size
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP only on CUDA
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()

        # Unfreeze
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            print(f"[INFO] Unfreezing backbone at epoch {epoch}")
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=1e-4
            )

        # ---- TRAIN ----
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)

            # Random temporal subset
            if args.sample_frames < clips.shape[1]:
                idxs = torch.randperm(clips.shape[1], device=clips.device)[:args.sample_frames]
                clips = clips.index_select(1, idxs).contiguous()

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(clips)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(clips)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples

        # ---- VALIDATION ----
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                if args.sample_frames < clips.shape[1]:
                    Tlen = clips.shape[1]
                    start_c = max(0, (Tlen - args.sample_frames)//2)
                    clips = clips[:, start_c:start_c+args.sample_frames].contiguous()
                logits = model(clips)
                all_preds.append(logits.argmax(1).cpu())
                all_labels.append(labels.cpu())
        all_preds = torch.cat(all_preds).numpy() if all_preds else np.array([])
        all_labels = torch.cat(all_labels).numpy() if all_labels else np.array([])

        # If val empty (edge case)
        if all_labels.size == 0:
            print("[WARN] Validation set is empty. Skipping metrics.")
            break

        # FORCE all classes into report
        labels_full = list(range(len(CLASS_NAMES)))
        report = classification_report(
            all_labels, all_preds,
            labels=labels_full,
            target_names=CLASS_NAMES,
            digits=3,
            zero_division=0
        )
        macro_f1 = f1_score(all_labels, all_preds,
                            labels=labels_full,
                            average="macro",
                            zero_division=0)

        elapsed = time.time() - start_t
        print(f"\nEpoch {epoch} | {elapsed:.1f}s | TrainLoss {train_loss:.4f} "
              f"Acc {train_acc:.3f} MacroF1 {macro_f1:.3f}")
        print(report)

        # Confusion matrix (full dimension)
        cm = confusion_matrix(all_labels, all_preds, labels=labels_full)
        print("Confusion Matrix (rows=true, cols=pred):")
        for r in cm:
            print("  ", r)

        # Early stopping
        if macro_f1 > best_f1 + 1e-4:
            best_f1 = macro_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_swin.pt")
            print("[INFO] Saved best_swin.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("[INFO] Early stopping (no macro-F1 improvement).")
                break

        scheduler.step()
        torch.save(model.state_dict(), f"swin_epoch{epoch}.pt")

if __name__ == "__main__":
    main()