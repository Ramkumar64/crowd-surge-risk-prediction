import pandas as pd
from collections import Counter

CLASS_NAMES = ["Normal","Light_Panic","Fight","Violent_Group","Stampede"]

def show(csv_path, title):
    df = pd.read_csv(csv_path)
    c = df.label.value_counts()
    print(f"\n== {title} ({csv_path}) ==")
    for cls in CLASS_NAMES:
        print(f"{cls:15s} {c.get(cls,0)}")
    print("Total:", len(df))
    # Per video
    pv = df.groupby(["video","label"]).size().reset_index(name="count")
    print("\nPer video (first 25 rows):")
    print(pv.head(25))

if __name__=="__main__":
    show("window_train.csv","TRAIN")
    show("window_val.csv","VAL")