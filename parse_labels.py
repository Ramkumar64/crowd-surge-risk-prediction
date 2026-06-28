import csv
import ast
import os

csv_file = '/Users/ithikash/Downloads/Crowd-Analysis-main/project-2-at-2025-09-27-13-10-2daef568.csv'

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"{csv_file} does not exist.")

segments = []
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_file = row["video"].split("/")[-1]
        try:
            videoLabels = ast.literal_eval(row["videoLabels"].replace('""', '"'))
        except Exception as e:
            print(f"Error parsing videoLabels for {video_file}: {e}")
            continue
        for segment in videoLabels:
            label = segment.get("timelinelabels", ["Unknown"])[0]
            for r in segment.get("ranges", []):
                start = r.get("start")
                end = r.get("end")
                segments.append({
                    "video": video_file,
                    "start_frame": start,
                    "end_frame": end,
                    "label": label
                })

print("First 10 labeled segments:")
for seg in segments[:10]:
    print(seg)
print(f"\nTotal labeled segments: {len(segments)}")