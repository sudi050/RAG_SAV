import pandas as pd
import numpy as np
import json

precedents_df = pd.read_json("il_pcsr/precedents.json", lines=True)

def to_list(val):
    if val is None: return []
    if isinstance(val, (list, np.ndarray)): return list(val)
    return []

precedent_statute_map = {}
for _, row in precedents_df.iterrows():
    pid    = str(row["id"])
    labels = list(set(str(s) for s in to_list(row["relevant_statute_ids"])))
    precedent_statute_map[pid] = labels

counts = pd.Series([len(v) for v in precedent_statute_map.values()])
print(f"Total precedents       : {len(precedent_statute_map)}")
print(f"With statute labels    : {(counts > 0).sum()} ({(counts>0).mean()*100:.1f}%)")
print(f"Without statute labels : {(counts == 0).sum()} ({(counts==0).mean()*100:.1f}%)")
print(f"Mean statutes/precedent: {counts.mean():.2f}")

with open("results/primary_analysis/precedent_statute_map.json", "w") as f:
    json.dump(precedent_statute_map, f)
print("✅ Saved precedent_statute_map.json")
