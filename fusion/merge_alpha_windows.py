import pandas as pd, glob, os

files = glob.glob("**/alpha_windows.csv", recursive=True)
if not files:
    print("No alpha_windows.csv found.")
    raise SystemExit

dfs=[]
for f in files:
    df=pd.read_csv(f)
    df["source_file"]=os.path.abspath(f)
    df["video_id"]=os.path.basename(os.path.dirname(f))
    dfs.append(df)

master=pd.concat(dfs, ignore_index=True)
master.to_csv("all_alpha_windows.csv", index=False)
print("Created all_alpha_windows.csv rows=", len(master), "from", len(files), "files.")