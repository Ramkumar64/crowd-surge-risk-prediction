import pandas as pd, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv).sort_values("start_proc_frame", ignore_index=True)

    for col in ["crowd_count","occupancy","mean_speed","risk_ema","risk_score"]:
        if col in df.columns:
            df[f"delta_{col}"] = df[col].diff().fillna(0.0)

    # Optional ratio feature
    if "mean_speed" in df.columns and "crowd_count" in df.columns:
        df["speed_per_person"] = df["mean_speed"] / (df["crowd_count"] + 1e-6)

    df.to_csv(args.out_csv, index=False)
    print("[SAVED]", args.out_csv, "rows=", len(df))

if __name__ == "__main__":
    main()