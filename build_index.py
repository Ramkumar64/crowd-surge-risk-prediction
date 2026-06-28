import os, csv, cv2

CLIPS_ROOT = "extracted_clips_clean"
OUT_CSV = "dataset_index.csv"
VALID_EXT = {".mp4"}

rows=[]
bad=[]
for label in sorted(os.listdir(CLIPS_ROOT)):
    ldir = os.path.join(CLIPS_ROOT, label)
    if not os.path.isdir(ldir): continue
    for clip_group in os.listdir(ldir):
        gdir = os.path.join(ldir, clip_group)
        if not os.path.isdir(gdir): continue
        for f in os.listdir(gdir):
            if not any(f.endswith(ext) for ext in VALID_EXT): continue
            path = os.path.join(gdir, f)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                bad.append(path); continue
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            if nframes <= 0:
                bad.append(path); continue
            rows.append({
                "clip_path": path,
                "label": label,
                "frames": nframes,
                "fps": fps,
                "duration_sec": f"{nframes/fps:.3f}"
            })

with open(OUT_CSV,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["clip_path","label","frames","fps","duration_sec"])
    w.writeheader()
    w.writerows(rows)

print(f"Indexed {len(rows)} clips into {OUT_CSV}")
if bad:
    print("Bad/Corrupt clips (remove manually if desired):")
    for b in bad:
        print(" -", b)