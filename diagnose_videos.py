import os, cv2, json

VIDEO_DIR = "/Users/ithikash/Downloads/Crowd-Analysis-main/HAJJv2.Dataset/Training/Videos"

# Map annotation names -> actual file names
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

# Import segments from your parse script (CSV-based)
from parse_labels import segments  # adjust if different

# Group annotation end frames per real video
per_video = {}
for seg in segments:
    ann_name = seg['video']
    real = FILENAME_MAP.get(ann_name, ann_name)
    per_video.setdefault(real, {"max_end": 0})
    if seg['end_frame'] > per_video[real]["max_end"]:
        per_video[real]["max_end"] = seg['end_frame']

scales = {}
for real, info in per_video.items():
    path = os.path.join(VIDEO_DIR, real)
    if not os.path.exists(path):
        print("Missing video:", path)
        continue
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    mae = info["max_end"]
    if mae <= 0 or total <= 1:
        scale = 1.0
    else:
        # If annotation numbers extend beyond real frames, compress them.
        scale = (total - 1) / mae if mae >= total else 1.0
    scales[real] = {
        "real_frames": total,
        "max_annot_end": mae,
        "scale_factor": scale,
        "needs_scaling": scale < 0.999999
    }

print("Video scaling diagnostics:")
for k,v in scales.items():
    print(f"{k}: real_frames={v['real_frames']} max_annot_end={v['max_annot_end']} scale_factor={v['scale_factor']:.6f} needs_scaling={v['needs_scaling']}")

with open("video_scaling_factors.json","w") as f:
    json.dump(scales, f, indent=2)
print("\nSaved scaling factors to video_scaling_factors.json")