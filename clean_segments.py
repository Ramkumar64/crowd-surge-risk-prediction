import json, os

from parse_labels import segments  # original raw segments (start_frame/end_frame as in annotation)

FILENAME_MAP = {
    "fb166138-Charlie_Kirk_shooting__Video_captures_panic_at_Utah_campus_as_gunshot_heard.mp4": "Charlie Kirk shooting_ Video captures panic at Utah campus as gunshot heard.mp4",
    "7d1ef852-2.mp4": "2.mp4",
    "c8d8b2b8-3.mp4": "3.mp4",
    "c2fdcaa1-5.mp4": "5.mp4",
    "5622a1ac-7.mp4": "7.mp4",
    "ab9189cd-8.mp4": "8.mp4",
    "598de69f-newyork.mp4": "newyork.mp4",
    "6fbb9aa1-ezyZip.mp4": "ezyZip.mp4",
    "7a8aaa94-ssvid.net---England-fans-fight-Scotland-fans-England-vs-Scotland-brawl_v720P.mp4": "ssvid.net---England-fans-fight-Scotland-fans-England-vs-Scotland-brawl_v720P.mp4",
    "aa197093-ssvid.net--Video-Stampede-storms-entrance-of-Astroworld-festival_v720P.mp4": "ssvid.net--Video-Stampede-storms-entrance-of-Astroworld-festival_v720P.mp4",
}

with open("video_scaling_factors.json","r") as f:
    scaling = json.load(f)

# Group per (video,label)
grouped = {}
for seg in segments:
    ann = seg['video']
    real = FILENAME_MAP.get(ann, ann)
    key = (real, seg['label'])
    grouped.setdefault(key, []).append({
        "orig_start": int(seg['start_frame']),
        "orig_end": int(seg['end_frame'])
    })

# Deduplicate: remove segments completely inside another
def merge_and_filter(elems):
    # Sort by start asc, length desc
    elems.sort(key=lambda x: (x["orig_start"], -(x["orig_end"] - x["orig_start"])))
    filtered=[]
    for i, e in enumerate(elems):
        contained=False
        for j, f in enumerate(elems):
            if j==i: continue
            if f["orig_start"] <= e["orig_start"] and f["orig_end"] >= e["orig_end"]:
                # e is inside f
                contained=True
                break
        if not contained:
            filtered.append(e)
    return filtered

clean_segments=[]
MIN_FRAMES = 4  # ensure at least this many frames after scaling

for (real,label), segs in grouped.items():
    filtered = merge_and_filter(segs)
    scale = scaling.get(real, {}).get("scale_factor", 1.0)
    real_total = scaling.get(real, {}).get("real_frames", None)
    for f in filtered:
        s_raw, e_raw = f["orig_start"], f["orig_end"]
        s_scaled = int(round(s_raw * scale))
        e_scaled = int(round(e_raw * scale))
        if real_total is not None:
            if s_scaled >= real_total: continue
            if e_scaled >= real_total:
                e_scaled = real_total - 1
        # Enforce ordering
        if e_scaled < s_scaled:
            continue
        # Enforce minimum length
        length = e_scaled - s_scaled + 1
        if length < MIN_FRAMES:
            deficit = MIN_FRAMES - length
            e_scaled += deficit
            if real_total is not None and e_scaled >= real_total:
                e_scaled = real_total - 1
                # Recompute length; if still < MIN_FRAMES and cannot extend, we can skip
                if (e_scaled - s_scaled + 1) < MIN_FRAMES:
                    continue
        clean_segments.append({
            "video_real": real,
            "label": label.replace(" ", "_"),
            "start_frame": s_scaled,
            "end_frame": e_scaled,
            "orig_start": s_raw,
            "orig_end": e_raw,
            "scale_used": scale
        })

print(f"Original segments: {len(segments)}  -> Clean usable segments: {len(clean_segments)}")

with open("clean_segments.json","w") as f:
    json.dump(clean_segments, f, indent=2)

print("Saved cleaned segments to clean_segments.json")