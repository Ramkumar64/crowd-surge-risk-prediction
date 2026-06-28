import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse, torch, torch.nn as nn, numpy as np, time
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import VideoMAEForVideoClassification, AutoImageProcessor, AutoConfig
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from dataset_videomae import VideoMAEWindowDataset, CLASS_NAMES
from focal_loss import FocalLoss

def collate(batch):
    frames_list, labels = zip(*batch)
    return list(frames_list), torch.tensor(labels)

def compute_alpha(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    inv = 1.0 / np.maximum(counts, 1)
    alpha = inv / inv.sum()
    return torch.tensor(alpha, dtype=torch.float32), counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="balanced_train.csv")
    ap.add_argument("--val_csv",   default="balanced_val.csv")
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--frames_per_clip", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--model_name", default="MCG-NJU/videomae-base")
    ap.add_argument("--use_focal", action="store_true")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--weighted_sampler", action="store_true")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=2)
    ap.add_argument("--no_freeze", action="store_true", help="Skip backbone freezing warmup.")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=0)  # default 0 for stability
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"[CONFIG] device={device} model={args.model_name} workers={args.num_workers}")

    # Dataset
    train_ds = VideoMAEWindowDataset(args.train_csv, args.video_root,
                                     frames_per_clip=args.frames_per_clip,
                                     train=True, seed=args.seed)
    val_ds   = VideoMAEWindowDataset(args.val_csv, args.video_root,
                                     frames_per_clip=args.frames_per_clip,
                                     train=False, seed=args.seed)

    # Sampler (optional)
    if args.weighted_sampler:
        labs = [lbl for _,lbl in train_ds]
        counts = np.bincount(labs, minlength=len(CLASS_NAMES))
        weights = 1.0 / np.maximum(counts,1)
        sample_w = [weights[l] for l in labs]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w)*2, replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers, collate_fn=collate,
                                  pin_memory=(device=="cuda"))
        print("[INFO] Weighted sampler counts:", counts)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate,
                                  pin_memory=(device=="cuda"))

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=(device=="cuda"))

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(CLASS_NAMES),
        label2id={c:i for i,c in enumerate(CLASS_NAMES)},
        id2label={i:c for i,c in enumerate(CLASS_NAMES)}
    )
    model = VideoMAEForVideoClassification.from_pretrained(args.model_name, config=config).to(device)

    def set_backbone(req):
        for n,p in model.named_parameters():
            if "classifier" in n: continue
            p.requires_grad = req

    if not args.no_freeze:
        set_backbone(False)

    train_labels = np.array([lbl for _,lbl in train_ds])
    alpha, counts = compute_alpha(train_labels, len(CLASS_NAMES))
    print("[INFO] Class counts:", counts)

    if args.use_focal:
        criterion = FocalLoss(alpha=alpha.to(device), gamma=args.gamma)
        print("[INFO] Using Focal Loss alpha:", alpha.numpy())
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha.to(device))
        print("[INFO] Using weighted CrossEntropy weights:", alpha.numpy())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    best_f1=0; patience=0

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        if (not args.no_freeze) and epoch == args.freeze_backbone_epochs + 1:
            print(f"[INFO] Unfreezing backbone at epoch {epoch}")
            set_backbone(True)

        # -------- Train --------
        model.train()
        tr_loss=0; tr_correct=0; tr_total=0
        for batch_idx, (frames_list, labels) in enumerate(train_loader):
            labels = labels.to(device)
            pixel_batches=[]
            for frames in frames_list:
                enc = processor(frames, return_tensors="pt")
                pixel_batches.append(enc["pixel_values"])
            pixel_values = torch.cat(pixel_batches, dim=0).to(device)

            optimizer.zero_grad()
            use_amp = (device=="cuda")
            try:
                if use_amp:
                    with torch.cuda.amp.autocast():
                        out = model(pixel_values=pixel_values)
                        loss = criterion(out.logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(pixel_values=pixel_values)
                    loss = criterion(out.logits, labels)
                    loss.backward()
                    optimizer.step()
            except RuntimeError as e:
                print("[WARN] RuntimeError in batch", batch_idx, ":", e)
                continue

            tr_loss += loss.item()*labels.size(0)
            tr_correct += (out.logits.argmax(1)==labels).sum().item()
            tr_total += labels.size(0)

            if (batch_idx+1) % args.log_every == 0:
                print(f"  Batch {batch_idx+1} | PartialLoss {(tr_loss/tr_total):.4f}")

        train_loss = tr_loss / max(1,tr_total)
        train_acc  = tr_correct / max(1,tr_total)

        # -------- Val --------
        model.eval()
        preds=[]; ytrue=[]
        with torch.no_grad():
            for frames_list, labels in val_loader:
                labels = labels.to(device)
                pixel_batches=[]
                for frames in frames_list:
                    enc = processor(frames, return_tensors="pt")
                    pixel_batches.append(enc["pixel_values"])
                pixel_values = torch.cat(pixel_batches, dim=0).to(device)
                logits = model(pixel_values=pixel_values).logits
                preds.append(logits.argmax(1).cpu())
                ytrue.append(labels.cpu())

        if preds:
            preds = torch.cat(preds).numpy()
            ytrue = torch.cat(ytrue).numpy()
            from sklearn.metrics import classification_report, confusion_matrix, f1_score
            rep = classification_report(ytrue, preds,
                                        labels=list(range(len(CLASS_NAMES))),
                                        target_names=CLASS_NAMES,
                                        digits=3, zero_division=0)
            macroF1 = f1_score(ytrue, preds, average="macro", zero_division=0)
            cm = confusion_matrix(ytrue, preds, labels=list(range(len(CLASS_NAMES))))
        else:
            rep = "NO VALIDATION BATCHES"
            macroF1 = 0
            cm = []

        elapsed = time.time()-t0
        print(f"\nEpoch {epoch} | {elapsed:.1f}s | TrainLoss {train_loss:.4f} Acc {train_acc:.3f} MacroF1 {macroF1:.3f}")
        print(rep)
        if len(cm):
            print("Confusion Matrix:")
            for r in cm: print(" ", r)

        if macroF1 > best_f1 + 1e-4:
            best_f1=macroF1
            patience=0
            torch.save(model.state_dict(), "best_videomae_stable.pt")
            print("[INFO] Saved best_videomae_stable.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print("[INFO] Early stopping.")
                break

        scheduler.step()
        torch.save(model.state_dict(), f"videomae_stable_epoch{epoch}.pt")

if __name__ == "__main__":
    main()