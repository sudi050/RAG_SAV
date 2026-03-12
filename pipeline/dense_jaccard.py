import pandas as pd
import numpy as np
import json

# ── Load ──────────────────────────────────────────────────────────────────
dev_df        = pd.read_json("il_pcsr/dev_queries.json",  lines=True)
test_df       = pd.read_json("il_pcsr/test_queries.json", lines=True)
precedents_df = pd.read_json("il_pcsr/precedents.json",   lines=True)

with open("results/maps/dense_scores.json") as f:
    all_dense_scores = json.load(f)

with open("results/primary_analysis/precedent_statute_map.json") as f:
    prec_statute_map = json.load(f)

DEV_OFFSET  = 5017
TEST_OFFSET = 5017 + 627
prec_ids    = precedents_df["id"].astype(str).tolist()

def to_list(val):
    if val is None: return []
    if isinstance(val, (list, np.ndarray)): return list(val)
    return []

def normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def evaluate_combined(df, offset, alpha):
    r1 = r5 = r10 = mrr = valid = 0
    for i, (_, row) in enumerate(df.iterrows()):
        relevant = set(str(p) for p in to_list(row["relevant_precedent_ids"]))
        if not relevant: continue
        q_stat = set(str(s) for s in to_list(row["relevant_statute_ids"]))
        dense  = normalize(np.array(all_dense_scores[offset + i]))
        jac    = normalize(np.array([
            jaccard(q_stat, set(str(x) for x in prec_statute_map.get(prec_ids[j], [])))
            for j in range(len(prec_ids))
        ]))
        combined   = alpha * dense + (1 - alpha) * jac
        ranked     = np.argsort(combined)[::-1]
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
    print(f"  α={alpha}  R@1={r1/n:.4f}  R@5={r5/n:.4f}  R@10={r10/n:.4f}  MRR={mrr/n:.4f}")
    return {"r1": r1/n, "r5": r5/n, "r10": r10/n, "mrr": mrr/n}

print("── DEV — Alpha sweep ────────────────────────────")
best_alpha, best_mrr = None, 0
for alpha in [0.9, 0.7, 0.5, 0.3]:
    res = evaluate_combined(dev_df, DEV_OFFSET, alpha)
    if res["mrr"] > best_mrr:
        best_mrr, best_alpha = res["mrr"], alpha

print(f"\nBest α on dev (by MRR): {best_alpha}")
print("\n── TEST — Best alpha ────────────────────────────")
evaluate_combined(test_df, TEST_OFFSET, best_alpha)
