import os, json, csv, math, random
import argparse
from collections import defaultdict

def load_config(path):
    with open(path,'r') as f:
        return json.load(f)

def main(cfg):
    random.seed(cfg.get("seed", 42))
    with open(cfg["clean_segments_file"], 'r') as f:
        clean_segments = json.load(f)
    # Optional: load scaling (not strictly needed—segments already scaled)
    if os.path.exists(cfg["scales_file"]):
        with open(cfg["scales_file"], 'r') as f:
            scale_meta = json.load(f)
    else:
        scale_meta = {}

    window_frames = cfg["window_frames"]
    stride = cfg["stride"]
    min_eff = cfg["min_effective_frames"]
    pad_short = cfg["pad_short"]
    class_list = cfg["class_list"]

    rows = []
    segment_id = 0

    for seg in clean_segments:
        video = seg["video_real"]
        label = seg["label"]  # already underscored
        start_f = seg["start_frame"]
        end_f = seg["end_frame"]
        length = end_f - start_f + 1
        if length < min_eff:
            continue

        if length < window_frames:
            if not pad_short:
                continue
            # single padded window
            rows.append({
                "video": video,
                "label": label,
                "win_start": start_f,
                "win_end": end_f,
                "orig_segment_start": seg["orig_start"],
                "orig_segment_end": seg["orig_end"],
                "segment_id": segment_id,
                "need_pad": 1
            })
        else:
            # sliding windows
            cur = start_f
            while cur + window_frames - 1 <= end_f:
                rows.append({
                    "video": video,
                    "label": label,
                    "win_start": cur,
                    "win_end": cur + window_frames - 1,
                    "orig_segment_start": seg["orig_start"],
                    "orig_segment_end": seg["orig_end"],
                    "segment_id": segment_id,
                    "need_pad": 0
                })
                cur += stride
            # tail window if remainder large enough but missed last chunk
            if end_f - (cur - stride) >= window_frames // 2 and (cur + window_frames - 1) > end_f:
                tail_start = end_f - window_frames + 1
                if tail_start >= start_f:
                    rows.append({
                        "video": video,
                        "label": label,
                        "win_start": tail_start,
                        "win_end": end_f,
                        "orig_segment_start": seg["orig_start"],
                        "orig_segment_end": seg["orig_end"],
                        "segment_id": segment_id,
                        "need_pad": 0
                    })
        segment_id += 1

    # Optional balancing (oversample by class)
    if cfg.get("balance_mode","none") == "oversample":
        by_class = defaultdict(list)
        for r in rows:
            by_class[r["label"]].append(r)
        max_len = max(len(v) for v in by_class.values())
        balanced = []
        for cls, items in by_class.items():
            if len(items) == 0:
                continue
            reps = max_len // len(items)
            rem = max_len % len(items)
            balanced.extend(items * reps)
            balanced.extend(random.choices(items, k=rem))
        rows = balanced
        random.shuffle(rows)

    # Shuffle final
    random.shuffle(rows)

    out_path = cfg["output_window_index"]
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            "video","label","win_start","win_end",
            "orig_segment_start","orig_segment_end",
            "segment_id","need_pad"
        ])
        w.writeheader()
        w.writerows(rows)

    # Simple stats
    counts = defaultdict(int)
    for r in rows:
        counts[r["label"]] += 1

    print(f"Generated {len(rows)} windows.")
    for c in class_list:
        print(f"{c}: {counts.get(c,0)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="window_config.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    main(cfg)