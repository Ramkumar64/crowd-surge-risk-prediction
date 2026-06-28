import argparse, pandas as pd, numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

"""
Evaluate different probability thresholds for a saved prediction file
OR after running model inference (probabilities vs true labels).

Expected CSV columns:
  prob   (model probability of positive)
  label  (0/1 actual)
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="CSV with prob,label columns")
    ap.add_argument("--steps", type=int, default=41)
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)
    if not {"prob","label"} <= set(df.columns):
        raise ValueError("pred_csv must contain columns: prob, label")

    y = df["label"].values
    p = df["prob"].values

    roc = roc_auc_score(y,p) if len(set(y))>1 else -1
    prauc = average_precision_score(y,p) if len(set(y))>1 else -1
    print(f"ROC_AUC={roc:.3f}  PR_AUC={prauc:.3f}")

    best_f1=0; best_thr=None; rows=[]
    for thr in np.linspace(0,1,args.steps):
        preds = (p >= thr).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        rows.append((thr,prec,rec,f1))
        if f1>best_f1:
            best_f1=f1; best_thr=thr
    print(f"BEST threshold={best_thr:.3f} F1={best_f1:.3f}")
    print("Top 10 thresholds by F1:")
    top = sorted(rows, key=lambda x: x[3], reverse=True)[:10]
    for t in top:
        print(f"thr={t[0]:.3f} prec={t[1]:.3f} rec={t[2]:.3f} f1={t[3]:.3f}")

if __name__ == "__main__":
    main()