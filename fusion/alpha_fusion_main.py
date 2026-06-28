#!/usr/bin/env python3
"""
alpha_fusion_main.py (with live delta features + gating)

Adds live computation of:
    delta_crowd_count
    delta_occupancy
    delta_mean_speed
    delta_risk_ema
    delta_risk_score
    speed_per_person

Also includes:
    - 2-window alert gating (consecutive requirement)
    - Optional tier logic
    - Optional predictor probability EMA (commented; enable if wanted)

If you retrain with new delta features, ensure names MATCH exactly.
"""

import os, cv2, math, time, json, argparse, numpy as np, pandas as pd, torch
import torch.nn as nn
from collections import deque

from detection_tracking import (
    DetectorTracker,
    direction_entropy,
    mean_pairwise_distance,
    velocity_histogram,
    density_grid
)
from activity_module import ActivityModule, ACTIVITY_CLASSES


# ---------------- MLP (must match training architecture) ----------------
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
    def forward(self, x):
        return self.net(x)


# ---------------- Helpers ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Alpha fusion with learned predictor + live delta features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--video", required=True)
    p.add_argument("--out_dir", default="alpha_fused")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--classes", default="0")
    p.add_argument("--imgsz_base", type=int, default=640)
    p.add_argument("--dual_scale", action="store_true")
    p.add_argument("--big_scale", type=int, default=896)
    p.add_argument("--tracker", choices=["bytetrack","deepsort"], default="bytetrack")
    p.add_argument("--max_age", type=int, default=30)
    p.add_argument("--n_init", type=int, default=3)
    p.add_argument("--activity_ckpt", default=None)
    p.add_argument("--activity_T", type=int, default=32)
    p.add_argument("--activity_stride", type=int, default=16)
    p.add_argument("--win_size", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--frame_skip", type=int, default=0)
    p.add_argument("--velocity_bins", type=int, default=8)
    p.add_argument("--density_grid", type=int, default=4)
    p.add_argument("--risk_weights", type=str, default="0.4,0.25,0.2,0.15")
    p.add_argument("--risk_clip", type=float, default=5.5)
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--heatmap", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_tiers", action="store_true", help="Disable tiered alert logic")
    p.add_argument("--no_gating", action="store_true", help="Disable 2-window gating of alerts")
    p.add_argument("--prob_ema", action="store_true", help="Enable probability EMA smoothing before gating")
    p.add_argument("--prob_ema_alpha", type=float, default=0.5, help="EMA alpha if --prob_ema used")
    return p.parse_args()


def auto_device(dev):
    if dev != "auto":
        return dev
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sigmoid(x): return 1/(1+math.exp(-x))


def overlay_heat(frame, centroids, radius=28):
    if not centroids:
        return frame
    h, w = frame.shape[:2]
    hm = np.zeros((h, w), dtype=np.float32)
    for (x, y) in centroids:
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(hm, (xi, yi), radius, 1, -1)
    hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=radius/2)
    hm_norm = hm / (hm.max() + 1e-6)
    hm_col = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.55, hm_col, 0.45, 0)


# ---------------- Main ----------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = auto_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Device: {device}")

    keep_classes = [int(c.strip()) for c in args.classes.split(",") if c.strip().isdigit()]
    risk_w = [float(x) for x in args.risk_weights.split(",")]
    if len(risk_w) != 4:
        raise ValueError("risk_weights needs 4 numbers")

    # Detector + tracking
    detector = DetectorTracker(
        model_path=args.model,
        device=device,
        conf=args.conf,
        classes=keep_classes,
        imgsz_base=args.imgsz_base,
        dual_scale=args.dual_scale,
        big_scale=args.big_scale,
        tracker_type=args.tracker,
        max_age=args.max_age,
        n_init=args.n_init
    )

    # Activity (optional)
    activity = ActivityModule(
        T=args.activity_T,
        stride=args.activity_stride,
        device=device,
        ckpt_path=args.activity_ckpt
    )

    # Predictor load
    predictor = None
    predictor_features = None
    predictor_threshold = None
    predictor_device = device
    for cand in [os.path.join(args.out_dir, "risk_predictor_config.json"),
                 "risk_predictor_config.json"]:
        if os.path.isfile(cand):
            try:
                with open(cand, "r") as f:
                    cfg = json.load(f)
                ckpt_path = cfg.get("model_path", "risk_predictor_v2.pt")
                predictor_threshold = float(cfg.get("threshold", 0.5))
                if not os.path.isfile(ckpt_path):
                    print(f"[WARN] Predictor ckpt '{ckpt_path}' missing; skipping.")
                    break
                ck = torch.load(ckpt_path, map_location="cpu")
                predictor_features = ck["features"]
                predictor = MLP(len(predictor_features))
                predictor.load_state_dict(ck["model"])
                predictor.to(predictor_device).eval()
                print(f"[INFO] Loaded predictor '{ckpt_path}' (thr={predictor_threshold})")
            except Exception as e:
                print("[WARN] Could not load predictor:", e)
            break

    # Video
    cap = cv2.VideoCapture(int(args.video)) if str(args.video).isdigit() else cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Cannot open video")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    print(f"[INFO] Video: {total_frames} frames @ {fps_in:.2f} fps {W}x{H}")

    writer = None
    if args.save_video:
        vid_path = os.path.join(args.out_dir, "annotated.mp4")
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
        if not writer.isOpened():
            print("[WARN] Could not open VideoWriter; disabling save.")
            writer = None

    per_frame_rows = []
    window_rows = []
    frame_buffer = []
    frame_idx = 0
    proc_idx = 0
    risk_ema_prev = None
    prev_window_row = None  # for delta features
    prob_ema_value = None
    recent_alerts = deque(maxlen=2)  # for 2-window gating (adjust maxlen for stricter gating)
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if 0 < args.max_frames <= frame_idx:
            break

        if args.frame_skip > 0 and (frame_idx % (args.frame_skip + 1)) != 0:
            frame_idx += 1
            continue

        act_probs = activity.step(frame)
        tracks = detector.get_tracks(proc_idx, frame)
        per_frame_rows.extend({"orig_frame": frame_idx, "proc_frame": proc_idx, **t} for t in tracks)
        frame_buffer.append(tracks)

        # Draw tracks
        for t in tracks:
            x1,y1,x2,y2 = map(int,[t["x1"],t["y1"],t["x2"],t["y2"]])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"ID{t['track_id']}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # Window condition
        if proc_idx >= args.win_size and ((proc_idx - args.win_size) % args.stride == 0):
            window_slice = frame_buffer[-args.win_size:]
            agg_tracks = {}
            centroids_all = []
            speeds_all = []
            vectors_all = []

            for fr in window_slice:
                for o in fr:
                    tid = o["track_id"]
                    agg_tracks.setdefault(tid, []).append((o["cx"], o["cy"]))
                    centroids_all.append((o["cx"], o["cy"]))

            for pts in agg_tracks.values():
                if len(pts) > 1:
                    for i in range(1, len(pts)):
                        dx = pts[i][0] - pts[i-1][0]
                        dy = pts[i][1] - pts[i-1][1]
                        vectors_all.append((dx, dy))
                        speeds_all.append(math.hypot(dx, dy))

            crowd_count = len(agg_tracks)
            mean_speed = float(np.mean(speeds_all)) if speeds_all else 0.0
            speed_std = float(np.std(speeds_all)) if speeds_all else 0.0
            median_speed = float(np.median(speeds_all)) if speeds_all else 0.0
            dir_ent = direction_entropy(vectors_all)
            mpd = mean_pairwise_distance(centroids_all)

            occupancy = 0.0
            last_frame_tracks = window_slice[-1]
            if last_frame_tracks:
                occupancy = sum((t["x2"]-t["x1"])*(t["y2"]-t["y1"]) for t in last_frame_tracks)/(W*H)

            v_hist = velocity_histogram(speeds_all, bins=args.velocity_bins)
            d_grid = density_grid(centroids_all, W, H, g=args.density_grid)
            act_stats = activity.stats_from_probs(act_probs)

            raw = (risk_w[0]*mean_speed +
                   risk_w[1]*min(occupancy*10,4) +
                   risk_w[2]*dir_ent*3 +
                   risk_w[3]*speed_std +
                   1.2*act_stats["act_abnormal_prob"])

            risk_score = sigmoid(raw / max(args.risk_clip, 1e-6))
            if risk_ema_prev is None:
                risk_ema = risk_score
            else:
                risk_ema = args.ema_alpha*risk_score + (1-args.ema_alpha)*risk_ema_prev
            risk_ema_prev = risk_ema

            row = {
                "start_proc_frame": proc_idx - args.win_size,
                "end_proc_frame": proc_idx - 1,
                "crowd_count": crowd_count,
                "occupancy": occupancy,
                "mean_speed": mean_speed,
                "median_speed": median_speed,
                "speed_std": speed_std,
                "dir_entropy": dir_ent,
                "mean_pairwise_distance": mpd,
                "risk_raw": raw,
                "risk_score": risk_score,
                "risk_ema": risk_ema
            }

            for i,cname in enumerate(ACTIVITY_CLASSES):
                row[f"act_p_{cname}"] = float(act_probs[i])
            row.update(act_stats)

            for i,v in enumerate(v_hist):
                row[f"vel_hist_{i}"] = float(v)

            flat_grid = d_grid.flatten()
            for i,v in enumerate(flat_grid):
                row[f"density_{i}"] = float(v)

            # -------- Live delta features (MATCH training names) --------
            if prev_window_row is not None:
                row["delta_crowd_count"] = row["crowd_count"] - prev_window_row.get("crowd_count",0.0)
                row["delta_occupancy"] = row["occupancy"] - prev_window_row.get("occupancy",0.0)
                row["delta_mean_speed"] = row["mean_speed"] - prev_window_row.get("mean_speed",0.0)
                row["delta_risk_ema"] = row["risk_ema"] - prev_window_row.get("risk_ema",0.0)
                row["delta_risk_score"] = row["risk_score"] - prev_window_row.get("risk_score",0.0)
            else:
                row["delta_crowd_count"] = 0.0
                row["delta_occupancy"] = 0.0
                row["delta_mean_speed"] = 0.0
                row["delta_risk_ema"] = 0.0
                row["delta_risk_score"] = 0.0

            row["speed_per_person"] = row["mean_speed"] / (row["crowd_count"] + 1e-6)

            # -------- Predictor inference + gating --------
            if predictor is not None:
                fv = [row.get(feat, 0.0) for feat in predictor_features]

                with torch.no_grad():
                    logits = predictor(
                        torch.tensor(fv, dtype=torch.float32).unsqueeze(0).to(predictor_device)
                    )
                    pred_prob = torch.softmax(logits,1)[0,1].item()

                # Optional probability EMA smoothing
                if args.prob_ema:
                    if prob_ema_value is None:
                        prob_ema_value = pred_prob
                    else:
                        prob_ema_value = (args.prob_ema_alpha * pred_prob +
                                          (1-args.prob_ema_alpha) * prob_ema_value)
                    row["pred_prob_ema"] = prob_ema_value
                    base_prob_for_decision = prob_ema_value
                else:
                    base_prob_for_decision = pred_prob

                raw_flag = int(base_prob_for_decision >= predictor_threshold)
                row["pred_prob"] = pred_prob
                row["pred_alert_raw"] = raw_flag

                if args.no_gating:
                    row["pred_alert"] = raw_flag
                else:
                    recent_alerts.append(raw_flag)
                    row["pred_alert"] = 1 if len(recent_alerts)==recent_alerts.maxlen and sum(recent_alerts)==recent_alerts.maxlen else 0
            else:
                row["pred_prob"] = None
                row["pred_alert_raw"] = 0
                row["pred_alert"] = 0

            # Tier logic (based on raw or final alert)
            if predictor is not None and not args.no_tiers:
                act_abn = row.get("act_abnormal_prob",0.0)
                violent_mix = row.get("act_p_Fight",0)+row.get("act_p_Violent_Group",0)
                riskE = row.get("risk_ema", row.get("risk_score",0))
                # Use raw_flag (immediate) for tier escalation
                tier = 0
                if row["pred_alert_raw"] == 1:
                    tier = 1
                if tier >= 1 and (violent_mix > 0.35 or act_abn > 0.55 or riskE > 0.65):
                    tier = 2
                if tier >= 1 and (violent_mix > 0.45 or act_abn > 0.65 or riskE > 0.75):
                    tier = 3
                row["alert_tier"] = tier
            else:
                row["alert_tier"] = 0

            window_rows.append(row)
            prev_window_row = row.copy()

            # --- Overlays ---
            top_cls = ACTIVITY_CLASSES[int(np.argmax(act_probs))]
            cv2.putText(frame,
                        f"{top_cls} abn={act_stats['act_abnormal_prob']:.2f}",
                        (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
            cv2.putText(frame,
                        f"RISK {risk_score:.2f} EMA {risk_ema:.2f}",
                        (10,40), cv2.FONT_HERSHEY_SIMPLEX,0.55,
                        (0,0,255) if risk_ema>0.65 else (0,255,255) if risk_ema>0.45 else (0,255,0),
                        2)

            if predictor is not None:
                disp_prob = row["pred_prob"] if row["pred_prob"] is not None else 0.0
                extra = f" raw={row['pred_alert_raw']} A={row['pred_alert']}"
                cv2.putText(frame,
                            f"PRED {disp_prob:.2f} thr={predictor_threshold:.2f}{extra}",
                            (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.5,
                            (0,0,255) if row["pred_alert"] else (200,200,200),2)
                if not args.no_tiers:
                    cv2.putText(frame,
                                f"TIER {row['alert_tier']}",
                                (10,80), cv2.FONT_HERSHEY_SIMPLEX,0.5,
                                (0,0,255) if row["alert_tier"]==3 else
                                (0,165,255) if row["alert_tier"]==2 else
                                (0,255,255) if row["alert_tier"]==1 else
                                (180,180,180),
                                2)

            if args.heatmap:
                frame = overlay_heat(frame, centroids_all)

        # Show / write
        if args.show:
            cv2.imshow("AlphaFusion", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if writer:
            writer.write(frame)

        frame_idx += 1
        proc_idx += 1

        if proc_idx % 150 == 0:
            elapsed = time.time()-t0
            print(f"[INFO] processed={proc_idx} ({proc_idx/elapsed:.2f} fps)")

    cap.release()
    if writer:
        writer.release()
        print(f"[SAVED] {os.path.join(args.out_dir,'annotated.mp4')}")

    # ----- Save outputs -----
    def to_py(v):
        import numpy as _np
        if isinstance(v, (_np.floating,_np.integer)): return v.item()
        if isinstance(v, _np.ndarray): return v.tolist()
        return v

    per_frame_csv = os.path.join(args.out_dir,"per_frame_tracks.csv")
    pd.DataFrame(per_frame_rows).to_csv(per_frame_csv, index=False)
    print("[SAVED]", per_frame_csv)

    if window_rows:
        windows_csv = os.path.join(args.out_dir,"alpha_windows.csv")
        pd.DataFrame(window_rows).to_csv(windows_csv, index=False)
        print("[SAVED]", windows_csv)

        windows_jsonl = os.path.join(args.out_dir,"alpha_windows.jsonl")
        with open(windows_jsonl,"w") as f:
            for r in window_rows:
                safe = {k: to_py(v) for k,v in r.items()}
                f.write(json.dumps(safe)+"\n")
        print("[SAVED]", windows_jsonl)
    else:
        print("[WARN] No windows generated.")

    print(f"[DONE] total_original_frames={frame_idx} processed_frames={proc_idx}")


if __name__ == "__main__":
    main()