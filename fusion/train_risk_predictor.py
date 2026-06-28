import argparse, pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_fscore_support
import random

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, dr=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hidden//2, 2)
        )
    def forward(self,x): return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma=gamma
        self.weight=weight
    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction='none')
        pt = torch.softmax(logits,1).gather(1, target.unsqueeze(1)).squeeze(1)
        fl = (1-pt)**self.gamma * ce
        return fl.mean()

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def select_features(df, target):
    ignore_exact = {target, "escalation_level","surge_label",
                    "escalation_level_heur","surge_label_heur",
                    "escalation_level_relaxed","surge_label_relaxed"}
    ignore_prefix = ("start_", "end_", "start_proc", "end_proc")
    feats=[]
    for c in df.columns:
        if c in ignore_exact: continue
        if any(c.startswith(p) for p in ignore_prefix): continue
        if df[c].dtype.kind not in "biufc": continue
        feats.append(c)
    return feats

def threshold_sweep(probs, labels, thr_list):
    results=[]
    for thr in thr_list:
        pred = (probs >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
        results.append((thr, prec, rec, f1))
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="surge_label")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos_weight", default="auto", help="'auto' or numeric (e.g. 4.0)")
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--out", default="risk_predictor_v2.pt")
    ap.add_argument("--show_sweep", action="store_true", help="Print threshold sweep each epoch")
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Target {args.target} not found in {args.csv}")

    feats = select_features(df, args.target)
    print(f"[INFO] Using {len(feats)} features: {feats[:6]}{'...' if len(feats)>6 else ''}")

    X = df[feats].values.astype(np.float32)
    y = df[args.target].values.astype(int)
    print("[INFO] Label distribution:", dict(zip(*np.unique(y, return_counts=True))))

    Xtr, Xval, ytr, yval = train_test_split(X,y,test_size=args.val_split,
                                            random_state=args.seed, stratify=y)
    print(f"[INFO] Train positives={ytr.sum()}  Val positives={yval.sum()}")

    model = MLP(len(feats)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Determine positive weight
    if args.pos_weight == "auto":
        pos = ytr.sum()
        neg = len(ytr)-pos
        pos_weight = (neg / max(pos,1))
    else:
        pos_weight = float(args.pos_weight)
    print(f"[INFO] pos_weight={pos_weight:.3f}")

    class_weight = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)

    if args.focal:
        criterion = FocalLoss(gamma=args.gamma, weight=class_weight)
        print(f"[INFO] Using FocalLoss gamma={args.gamma}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight)

    def run_epoch(train=True):
        model.train(train)
        Xd, yd = (Xtr, ytr) if train else (Xval, yval)
        idx = np.arange(len(Xd))
        if train: np.random.shuffle(idx)
        total=0; probs_list=[]; labs=[]
        for i in range(0,len(idx),args.batch):
            b = idx[i:i+args.batch]
            xb = torch.tensor(Xd[b]).to(device)
            yb = torch.tensor(yd[b]).to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()*len(b)
            probs = torch.softmax(logits,1)[:,1].detach().cpu().numpy()
            probs_list.append(probs)
            labs.append(yb.cpu().numpy())
        probs = np.concatenate(probs_list); labs = np.concatenate(labs)
        hard = (probs >= 0.5).astype(int)
        rep = classification_report(labs, hard, output_dict=True, zero_division=0)
        f1_pos = rep.get("1",{}).get("f1-score",0.0)
        return total/len(idx), probs, labs, f1_pos

    best_metric = -1
    for ep in range(1, args.epochs+1):
        tr_loss, tr_probs, tr_y, tr_f1 = run_epoch(True)
        val_loss, val_probs, val_y, val_f1 = run_epoch(False)

        # Sweep thresholds 0.2..0.6 step 0.1
        if args.show_sweep:
            sweep = threshold_sweep(val_probs, val_y, [0.2,0.3,0.4,0.5,0.6])
            sweep_str = " ".join(f"{thr:.1f}:{f1:.2f}" for thr,_,__,f1 in sweep)
            print(f"Epoch {ep:02d} | TrLoss {tr_loss:.4f} | ValLoss {val_loss:.4f} | F1@0.5 {val_f1:.3f} | Sweep {sweep_str}")
        else:
            print(f"Epoch {ep:02d} | TrLoss {tr_loss:.4f} | ValLoss {val_loss:.4f} | F1@0.5 {val_f1:.3f}")

        # Use PR AUC or best F1@0.5 to decide saving
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(val_y, val_probs) if len(np.unique(val_y))>1 else 0
        metric = pr_auc  # could combine
        if metric > best_metric:
            best_metric = metric
            torch.save({"model": model.state_dict(),
                        "features": feats,
                        "type": "mlp"}, args.out)
            print(f"[INFO] Saved {args.out} (PR_AUC={pr_auc:.3f})")

    # Always save validation probabilities
    import pandas as pd
    pd.DataFrame({"prob": val_probs, "label": val_y}).to_csv("preds_val_v2.csv", index=False)
    print("[SAVED] preds_val_v2.csv")

    if len(np.unique(val_y))>1:
        roc = roc_auc_score(val_y, val_probs)
        pr  = average_precision_score(val_y, val_probs)
        print(f"Final ROC_AUC={roc:.3f}  PR_AUC={pr:.3f}")

if __name__ == "__main__":
    main()