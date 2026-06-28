import os, json, subprocess, math

VIDEO_DIR = "/Users/ithikash/Downloads/Crowd-Analysis-main/HAJJv2.Dataset/Training/Videos"
OUTPUT_DIR = "extracted_clips_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("clean_segments.json","r") as f:
    segments = json.load(f)

def probe_fps(path):
    cmd = ["ffprobe","-v","error","-select_streams","v:0",
           "-show_entries","stream=avg_frame_rate",
           "-of","default=nokey=1:noprint_wrappers=1", path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return 30.0
    rate = p.stdout.strip()
    if "/" in rate:
        num, den = rate.split("/")
        try:
            return float(num)/float(den)
        except:
            return 30.0
    try:
        return float(rate)
    except:
        return 30.0

MIN_FRAMES = 4

for seg in segments:
    real = seg["video_real"]
    label = seg["label"]
    start_f = seg["start_frame"]
    end_f = seg["end_frame"]
    src = os.path.join(VIDEO_DIR, real)
    if not os.path.exists(src):
        print("Missing src:", src)
        continue
    fps = probe_fps(src)
    duration_frames = end_f - start_f + 1
    if duration_frames < MIN_FRAMES:
        duration_frames = MIN_FRAMES
    start_sec = start_f / fps
    dur_sec = duration_frames / fps + 1e-4  # epsilon
    out_dir = os.path.join(OUTPUT_DIR, label, os.path.splitext(real)[0])
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{os.path.splitext(real)[0]}_{label}_orig{seg['orig_start']}-{seg['orig_end']}_scaled{start_f}-{end_f}.mp4"
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        print("Exists:", out_path)
        continue
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error",
        "-ss", f"{start_sec:.4f}",
        "-i", src,
        "-t", f"{dur_sec:.4f}",
        "-c:v","libx264","-preset","veryfast","-crf","23",
        "-an",
        out_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL:", out_path, "ERR:", r.stderr.strip())
    else:
        print("OK:", out_path)