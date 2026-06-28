import pandas as pd, argparse, random
from collections import Counter

CLASS_LIST = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", default="window_index.csv")
    ap.add_argument("--train_csv", default="balanced_train.csv")
    ap.add_argument("--val_csv", default="balanced_val.csv")
    ap.add_argument("--val_ratio", type=float, default=0.20)
    ap.add_argument("--min_val_per_class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.index_csv)
    random.seed(args.seed)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    val_idxs = set()
    grp = df.groupby("label")
    # Guarantee some per class
    for c in CLASS_LIST:
        block = grp.get_group(c) if c in grp.groups else None
        if block is None or block.empty:
            print(f"[WARN] Class {c} empty globally!")
            continue
        take = min(args.min_val_per_class, len(block))
        chosen = block.sample(take, random_state=args.seed).index.tolist()
        val_idxs.update(chosen)

    target_val = int(len(df)*args.val_ratio)
    if len(val_idxs) < target_val:
        remaining = [i for i in df.index if i not in val_idxs]
        # Weighted by inverse class frequency
        class_counts = df.label.value_counts().to_dict()
        weights = []
        for idx in remaining:
            weights.append(1.0 / class_counts[df.at[idx,"label"]])
        total_w = sum(weights)
        weights = [w/total_w for w in weights]
        needed = target_val - len(val_idxs)
        more = random.choices(remaining, weights=weights, k=needed)
        val_idxs.update(more)

    val_df = df.loc[list(val_idxs)]
    train_df = df.drop(list(val_idxs))

    print("Validation distribution:")
    print(val_df.label.value_counts())
    print("\nTrain distribution:")
    print(train_df.label.value_counts())

    train_df.to_csv(args.train_csv, index=False)
    val_df.to_csv(args.val_csv, index=False)
    print(f"Wrote {args.train_csv} ({len(train_df)}) and {args.val_csv} ({len(val_df)})")

if __name__ == "__main__":
    main()