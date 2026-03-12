import pandas as pd
import numpy as np
import json

# ── Load ──────────────────────────────────────────────────────────────────
dev_df        = pd.read_json("il_pcsr/dev_queries.json",  lines=True)
test_df       = pd.read_json("il_pcsr/test_queries.json", lines=True)
precedents_df = pd.read_json("il_pcsr/precedents.json",   lines=True)

with open("results/maps/dense_scores.json") as f:
    all_dense_scores = json.load(f)

DEV_OFFSET  = 5017        # queries 5017–5643
TEST_OFFSET = 5017 + 627  # queries 5644–6270
prec_ids    = precedents_df["id"].astype(str).tolist()

def to_list(val):
    if val is None: return []
    if isinstance(val, (list, np.ndarray)): return list(val)
    return []

def evaluate(df, offset, label):
    r1 = r5 = r10 = mrr = valid = 0
    for i, (_, row) in enumerate(df.iterrows()):
        relevant = set(str(p) for p in to_list(row["relevant_precedent_ids"]))
        if not relevant: continue
        scores   = np.array(all_dense_scores[offset + i])
        ranked   = np.argsort(scores)[::-1]
        ranked_ids = [prec_ids[j] for j in ranked]
        r1  += int(any(p in relevant for p in ranked_ids[:1]))
        r5  += int(any(p in relevant for p in ranked_ids[:5]))
        r10 += int(any(p in relevant for p in ranked_ids[:10]))
        for rank, pid in enumerate(ranked_ids[:100], 1):
            if pid in relevant:
                mrr += 1 / rank
                break
        valid += 1
    n = valid
    print(f"{label}  R@1={r1/n:.4f}  R@5={r5/n:.4f}  R@10={r10/n:.4f}  MRR={mrr/n:.4f}")

print("── Dense Baseline ───────────────────────────────")
evaluate(dev_df,  DEV_OFFSET,  "Dev ")
evaluate(test_df, TEST_OFFSET, "Test")
