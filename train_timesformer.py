import argparse, torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
from dataset_timeformer import TimeWindowDataset, CLASS_NAMES
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from transformers import TimeSformerForVideoClassification, AutoConfig
import time, os

def collate(batch):
    clips, labels = zip(*batch)
    clips = torch.stack(clips, dim=0)  # (B,T,C,H,W)
    labels = torch.tensor(labels)
    return clips, labels

def compute_class_weights(train_labels, num_classes):
    counts = np.bincount(train_labels, minlength=num_classes)
    inv = 1.0 / np.maximum(counts, 1)
    weights = inv / inv.sum()
    return torch.tensor(weights, dtype=torch.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="balanced_train.csv")
    ap.add_argument("--val_csv",   default="balanced_val.csv")
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--frames_per_clip", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--model_name", default="facebook/timesformer-base-finetuned-k400")
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--weighted_ce", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"[CONFIG] device={device}, model={args.model_name}")

    train_ds = TimeWindowDataset(args.train_csv, args.video_root,
                                 frames_per_clip=args.frames_per_clip,
                                 image_size=args.img_size, train=True, seed=args.seed)
    val_ds   = TimeWindowDataset(args.val_csv, args.video_root,
                                 frames_per_clip=args.frames_per_clip,
                                 image_size=args.img_size, train=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=(device=="cuda"))

    num_classes = len(CLASS_NAMES)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        label2id={c:i for i,c in enumerate(CLASS_NAMES)},
        id2label={i:c for i,c in enumerate(CLASS_NAMES)}
    )
    model = TimeSformerForVideoClassification.from_pretrained(
        args.model_name, config=config
    ).to(device)

    # Replace classification head (HF sets it already but to be explicit):
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes).to(device)

    train_labels_all = []
    for _, lbl in train_ds:
        train_labels_all.append(lbl)
    train_labels_all = np.array(train_labels_all)

    if args.weighted_ce and not args.focal:
        class_weights = compute_class_weights(train_labels_all, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("[INFO] Using weighted cross-entropy:", class_weights.cpu().numpy())
    elif args.focal:
        # FocalLoss implementation inline
        class_weights = compute_class_weights(train_labels_all, num_classes).to(device)
        print("[INFO] Focal alpha:", class_weights.cpu().numpy())
        class FocalLoss(nn.Module):
            def __init__(self, alpha, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
            def forward(self, logits, targets):
                ce = nn.functional.cross_entropy(logits, targets,
                                                 weight=self.alpha,
                                                 reduction='none')
                pt = torch.exp(-ce)
                loss = ((1-pt)**self.gamma) * ce
                return loss.mean()
        criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        tr_loss=0; tr_correct=0; tr_total=0
        for clips, labels in train_loader:
            # TimeSformer expects pixel_values: (B, T, C, H, W)
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=clips)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * labels.size(0)
            tr_correct += (logits.argmax(1)==labels).sum().item()
            tr_total += labels.size(0)
        train_loss = tr_loss / tr_total
        train_acc  = tr_correct / tr_total

        # Validation
        model.eval()
        preds=[]; ytrue=[]
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                outputs = model(pixel_values=clips)
                logits = outputs.logits
                preds.append(logits.argmax(1).cpu())
                ytrue.append(labels.cpu())
        preds = torch.cat(preds).numpy()
        ytrue = torch.cat(ytrue).numpy()

        labels_all = list(range(num_classes))
        rep = classification_report(ytrue, preds, labels=labels_all,
                                    target_names=CLASS_NAMES,
                                    digits=3, zero_division=0)
        macroF1 = f1_score(ytrue, preds, labels=labels_all,
                           average="macro", zero_division=0)
        cm = confusion_matrix(ytrue, preds, labels=labels_all)

        elapsed = time.time()-t0
        print(f"\nEpoch {epoch} | {elapsed:.1f}s | TrainLoss {train_loss:.4f} "
              f"Acc {train_acc:.3f} MacroF1 {macroF1:.3f}")
        print(rep)
        print("Confusion Matrix:")
        for r in cm: print(" ", r)

        if macroF1 > best_f1 + 1e-4:
            best_f1 = macroF1
            patience_counter = 0
            torch.save(model.state_dict(), "best_timesformer.pt")
            print("[INFO] Saved best_timesformer.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("[INFO] Early stopping.")
                break

        scheduler.step()
        torch.save(model.state_dict(), f"timesformer_epoch{epoch}.pt")

if __name__ == "__main__":
    main()