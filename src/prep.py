import argparse, os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

p = argparse.ArgumentParser(); p.add_argument("--out"); a = p.parse_args()
df = load_iris(as_frame=True).frame
tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df.target)
os.makedirs(a.out, exist_ok=True)
tr.to_csv(f"{a.out}/train.csv", index=False)
te.to_csv(f"{a.out}/test.csv", index=False)
