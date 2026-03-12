# pip install rank_bm25
import pandas as pd
import numpy as np
import json
from rank_bm25 import BM25Okapi
from collections import defaultdict

train_df = pd.read_json("il_pcsr/train_queries.json", lines=True)
dev_df   = pd.read_json("il_pcsr/dev_queries.json",   lines=True)
test_df  = pd.read_json("il_pcsr/test_queries.json",  lines=True)
train_df["split"] = "train"
dev_df["split"]   = "dev"
test_df["split"]  = "test"
queries_df    = pd.concat([train_df, dev_df, test_df], ignore_index=True)
precedents_df = pd.read_json("il_pcsr/precedents.json", lines=True)

def to_list(val):
    if val is None: return []
    if isinstance(val, (list, np.ndarray)): return list(val)
    return []

print("Building BM25 index...")
prec_ids   = precedents_df["id"].astype(str).tolist()
corpus     = [str(row["text"]).lower().split() for _, row in precedents_df.iterrows()]
bm25       = BM25Okapi(corpus)
prec_index = {pid: i for i, pid in enumerate(prec_ids)}
print(f"Index built — {len(prec_ids)} precedents")

def evaluate(queries, bm25, prec_ids, scores_override=None, label="BM25"):
    recall_1 = recall_5 = recall_10 = mrr = 0
    valid    = 0

    for i, (_, row) in enumerate(queries.iterrows()):
        if i % 100 == 0: print(f"  {i}/{len(queries)}...")
        relevant = set(str(p) for p in to_list(row["relevant_precedent_ids"]))
        if not relevant: continue

        if scores_override is not None:
            scores = scores_override[i]
        else:
            query_tokens = str(row["text"]).lower().split()
            scores       = bm25.get_scores(query_tokens)

        ranked = np.argsort(scores)[::-1]
        ranked_ids = [prec_ids[j] for j in ranked]

        # Recall@K
        recall_1  += int(any(pid in relevant for pid in ranked_ids[:1]))
        recall_5  += int(any(pid in relevant for pid in ranked_ids[:5]))
        recall_10 += int(any(pid in relevant for pid in ranked_ids[:10]))

        # MRR
        for rank, pid in enumerate(ranked_ids[:100], 1):
            if pid in relevant:
                mrr += 1 / rank
                break
        valid += 1

    n = valid
    print(f"\n── {label} RESULTS ({n} queries) ────────────────")
    print(f"  Recall@1  : {recall_1/n:.4f}")
    print(f"  Recall@5  : {recall_5/n:.4f}")
    print(f"  Recall@10 : {recall_10/n:.4f}")
    print(f"  MRR       : {mrr/n:.4f}")
    return {"recall@1": recall_1/n, "recall@5": recall_5/n,
            "recall@10": recall_10/n, "mrr": mrr/n}

print("\nEvaluating on DEV split...")
dev_results  = evaluate(dev_df, bm25, prec_ids, label="BM25 Dev")

print("\nEvaluating on TEST split...")
test_results = evaluate(test_df, bm25, prec_ids, label="BM25 Test")

print("\nSaving BM25 scores for all queries...")
all_bm25_scores = []
for i, (_, row) in enumerate(queries_df.iterrows()):
    if i % 200 == 0: print(f"  {i}/{len(queries_df)}...")
    tokens = str(row["text"]).lower().split()
    all_bm25_scores.append(bm25.get_scores(tokens).tolist())

with open("results/maps/bm25_scores.json", "w") as f:
    json.dump(all_bm25_scores, f)

results = {"dev": dev_results, "test": test_results}
with open("results/maps/bm25_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved bm25_scores.json and bm25_results.json")
