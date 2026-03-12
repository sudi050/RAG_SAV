import pandas as pd
import numpy as np
import json
from collections import Counter

train_df = pd.read_json("il_pcsr/train_queries.json", lines=True)
dev_df   = pd.read_json("il_pcsr/dev_queries.json",   lines=True)
test_df  = pd.read_json("il_pcsr/test_queries.json",  lines=True)
train_df["split"] = "train"
dev_df["split"]   = "dev"
test_df["split"]  = "test"
queries_df    = pd.concat([train_df, dev_df, test_df], ignore_index=True)
precedents_df = pd.read_json("il_pcsr/precedents.json", lines=True)
statutes_df   = pd.read_json("il_pcsr/statutes.json",   lines=True)

def to_list(val):
    if val is None: return []
    if isinstance(val, (list, np.ndarray)): return list(val)
    return []

# ── 1. Statute count distribution (citations vs annotators) ──────────────
q_stat_counts = queries_df["relevant_statutes"].apply(lambda x: len(to_list(x)))
p_stat_counts = precedents_df["relevant_statutes"].apply(lambda x: len(to_list(x)))

print("── RELEVANT STATUTES PER QUERY ─────────────────")
print(q_stat_counts.describe().round(2))
print(f"\n  Zero statutes (queries)    : {(q_stat_counts == 0).sum()}")
print(f"  Zero statutes (precedents) : {(p_stat_counts == 0).sum()}")

print("\n── RELEVANT STATUTES PER PRECEDENT ─────────────")
print(p_stat_counts.describe().round(2))

# ── 2. Are query statutes a subset of the statute pool? ──────────────────
print("\n── STATUTE POOL COVERAGE CHECK ─────────────────")
statute_pool_ids = set(statutes_df["id"].astype(str).tolist())
all_cited = []
out_of_pool = []

for val in queries_df["relevant_statute_ids"].dropna():
    ids = [str(i) for i in to_list(val)]
    all_cited.extend(ids)
    out_of_pool.extend([i for i in ids if i not in statute_pool_ids])

print(f"  Total statute citations     : {len(all_cited)}")
print(f"  Unique statutes cited       : {len(set(all_cited))}")
print(f"  Statute pool size           : {len(statute_pool_ids)}")
print(f"  Citations IN pool           : {len(all_cited) - len(out_of_pool)} ({(1-len(out_of_pool)/len(all_cited))*100:.1f}%)")
print(f"  Citations OUT of pool       : {len(out_of_pool)} ({len(out_of_pool)/len(all_cited)*100:.1f}%)")

# ── 3. Statute frequency (IDF signal) ────────────────────────────────────
print("\n── TOP 20 MOST CITED STATUTES (frequency) ──────")
freq = Counter(all_cited)
for sid, count in freq.most_common(20):
    name = statutes_df[statutes_df["id"].astype(str) == sid]["provision_name"].values
    label = name[0] if len(name) > 0 else "NOT IN POOL"
    print(f"  {sid:<10} {count:>5}x   {label[:60]}")

# ── 4. Statute overlap between queries and their cited precedents ─────────
print("\n── STATUTE OVERLAP: query vs cited precedents ───")
prec_statute_map = {
    str(row["id"]): set(str(s) for s in to_list(row["relevant_statute_ids"]))
    for _, row in precedents_df.iterrows()
}

overlaps = []
for _, row in queries_df.iterrows():
    q_stats  = set(str(s) for s in to_list(row["relevant_statute_ids"]))
    cited_p  = [str(p) for p in to_list(row["relevant_precedent_ids"])]
    if not q_stats or not cited_p:
        continue
    for pid in cited_p:
        if pid in prec_statute_map:
            p_stats  = prec_statute_map[pid]
            if not p_stats: continue
            inter    = q_stats & p_stats
            union    = q_stats | p_stats
            jaccard  = len(inter) / len(union) if union else 0
            overlaps.append(jaccard)

overlaps_s = pd.Series(overlaps)
print(f"  Query-precedent pairs checked : {len(overlaps)}")
print(f"  Mean Jaccard overlap          : {overlaps_s.mean():.3f}")
print(f"  Median Jaccard overlap        : {overlaps_s.median():.3f}")
print(f"  Pairs with zero overlap       : {(overlaps_s == 0).sum()} ({(overlaps_s==0).mean()*100:.1f}%)")
print(f"  Pairs with >0.5 overlap       : {(overlaps_s > 0.5).sum()} ({(overlaps_s>0.5).mean()*100:.1f}%)")
print(f"\n  Distribution:")
print(overlaps_s.describe().round(3))

# ── 5. Rhetorical role schema inspection ────────────────────────────────
print("\n── RHETORICAL ROLE SCHEMA ───────────────────────")
sample_roles = queries_df["rhetorical_roles"].dropna().iloc[0]
print(f"  Type of field : {type(sample_roles)}")
print(f"  First entry   : {sample_roles[0] if isinstance(sample_roles, (list, np.ndarray)) else sample_roles}")

all_roles = []
for val in queries_df["rhetorical_roles"].dropna():
    roles = to_list(val)
    if roles and isinstance(roles[0], dict):
        all_roles.extend([r.get("label") or r.get("role") or str(r) for r in roles])
    elif roles:
        all_roles.extend(roles)

role_counts = Counter(all_roles)
print(f"\n  Unique role labels: {len(role_counts)}")
print(f"  Role distribution:")
for role, count in role_counts.most_common():
    print(f"    {role:<40} {count:>8}")
