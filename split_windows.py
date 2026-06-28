import pandas as pd
import argparse, random, os

def main(index_csv, out_train="window_train.csv", out_val="window_val.csv",
         val_ratio=0.2, seed=42):
    random.seed(seed)
    df = pd.read_csv(index_csv)
    videos = sorted(df['video'].unique())
    random.shuffle(videos)
    n_val = max(1, int(len(videos)*val_ratio))
    val_set = set(videos[:n_val])
    train_rows = df[~df['video'].isin(val_set)]
    val_rows = df[df['video'].isin(val_set)]
    train_rows.to_csv(out_train, index=False)
    val_rows.to_csv(out_val, index=False)
    print(f"Videos total={len(videos)} train_videos={len(videos)-n_val} val_videos={n_val}")
    print(f"Train samples={len(train_rows)}  Val samples={len(val_rows)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="window_index.csv")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.index, val_ratio=args.val_ratio, seed=args.seed)