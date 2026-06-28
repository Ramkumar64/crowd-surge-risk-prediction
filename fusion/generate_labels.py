import argparse, pandas as pd, numpy as np, os

"""
Adds heuristic labels if manual labels not yet provided.

If your CSV already has 'escalation_level' or 'surge_label', it will not overwrite
unless you pass --force.

Usage:
  python generate_labels.py --in all_alpha_windows.csv --out all_alpha_windows_labeled.csv
"""

def add_heuristics(df):
    # Ensure needed columns exist
    needed_probs = ["act_p_Normal","act_p_Light_Panic","act_p_Fight","act_p_Violent_Group","act_p_Stampede"]
    for c in needed_probs:
        if c not in df.columns:
            # If missing (e.g. no activity model) fallback to Normal=1 others=0
            for cc in needed_probs:
                df[cc] = df.get(cc, 0.0)
            df["act_p_Normal"] = 1.0
            break

    # Derive abnormal prob if missing
    if "act_abnormal_prob" not in df.columns:
        df["act_abnormal_prob"] = 1.0 - df["act_p_Normal"].clip(0,1)

    # Apply heuristic rules
    esc = []
    for _, r in df.iterrows():
        stamp = r.get("act_p_Stampede", 0)
        fight = r.get("act_p_Fight", 0)
        violent = r.get("act_p_Violent_Group", 0)
        light = r.get("act_p_Light_Panic", 0)
        abn = r.get("act_abnormal_prob", 0)
        risk_ema = r.get("risk_ema", r.get("risk_score", 0))
        crowd = r.get("crowd_count", 0)

        if stamp >= 0.40 or (risk_ema >= 0.80 and abn >= 0.60):
            esc.append(3)
        elif (fight + violent) >= 0.45 or (risk_ema >= 0.65 and crowd > 25):
            esc.append(2)
        elif light >= 0.40 or risk_ema >= 0.55:
            esc.append(1)
        else:
            esc.append(0)
    df["escalation_level_heur"] = esc
    df["surge_label_heur"] = (df["escalation_level_heur"] >= 2).astype(int)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--force", action="store_true", help="Overwrite existing labels")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    already = any(c in df.columns for c in ["escalation_level","surge_label"])
    if already and not args.force:
        print("[INFO] Labels already present; writing copy with no changes.")
    else:
        df = add_heuristics(df)
        if "escalation_level" not in df.columns:
            df["escalation_level"] = df["escalation_level_heur"]
        if "surge_label" not in df.columns:
            df["surge_label"] = df["surge_label_heur"]

    df.to_csv(args.outp, index=False)
    print(f"[SAVED] {args.outp} rows={len(df)}")
    print(df[["escalation_level","surge_label"]].head())

if __name__ == "__main__":
    main()