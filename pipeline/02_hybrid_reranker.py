import pandas as pd
import numpy as np
import json

train_df = pd.read_json("il_pcsr/train_queries.json", lines=True)
dev_df   = pd.read_json("il_pcsr/dev_queries.json",   lines=True)
test_df  = pd.read_json("il_pcsr/test_queries.json",  lines=True)
train_df["split"] = "train"
dev_df["split"]   = "dev"
test_df["split"]  = "test"
queries_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
queries_df = queries_df.reset_index(drop=True)

precedents_df = pd.read_json("il_pcsr/precedents.json", lines=True)

with open("results/primary_analysis/precedent_statute_map.json") as f:
    prec_statute_map = json.load(f)
with open("results/maps/bm25_scores.json") as f:
    all_bm25_scores = json.load(f)

prec_ids = precedents_df["id"].astype(str).tolist()

def to_list(val):
    if val is None: return []
    if isinstance(val, (list, np.ndarray)): return list(val)
    return []

def jaccard(set_a, set_b):
    if not set_a or not set_b: return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0

def normalize(scores):
    mn, mx = min(scores), max(scores)
    if mx == mn: return [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]

print("Precomputing Jaccard scores...")
all_jaccard_scores = []

for i, (_, row) in enumerate(queries_df.iterrows()):
    if i % 200 == 0: print(f"  {i}/{len(queries_df)}...")
    q_statutes = set(str(s) for s in to_list(row["relevant_statute_ids"]))
    bm25_scores = np.array(all_bm25_scores[i])

    top50_idx = np.argsort(bm25_scores)[::-1][:50]

    jac_scores = np.zeros(len(prec_ids))
    for idx in top50_idx:
        pid = prec_ids[idx]
        p_statutes = set(prec_statute_map.get(pid, []))
        jac_scores[idx] = jaccard(q_statutes, p_statutes)

    all_jaccard_scores.append(jac_scores.tolist())

print("Done computing Jaccard scores.")

def evaluate_combined(queries, query_global_indices, alpha, label):
    recall_1 = recall_5 = recall_10 = mrr = valid = 0

    for local_i, (_, row) in enumerate(queries.iterrows()):
        global_i = query_global_indices[local_i]
        relevant = set(str(p) for p in to_list(row["relevant_precedent_ids"]))
        if not relevant: continue

        bm25_s = np.array(all_bm25_scores[global_i])
        jac_s  = np.array(all_jaccard_scores[global_i])

        bm25_norm = np.array(normalize(bm25_s.tolist()))
        jac_norm  = np.array(normalize(jac_s.tolist()))

        combined  = alpha * bm25_norm + (1 - alpha) * jac_norm
        ranked    = np.argsort(combined)[::-1]
        ranked_ids = [prec_ids[j] for j in ranked]

        recall_1  += int(any(pid in relevant for pid in ranked_ids[:1]))
        recall_5  += int(any(pid in relevant for pid in ranked_ids[:5]))
        recall_10 += int(any(pid in relevant for pid in ranked_ids[:10]))
        for rank, pid in enumerate(ranked_ids[:100], 1):
            if pid in relevant:
                mrr += 1 / rank
                break
        valid += 1

    n = valid
    r = {"alpha": alpha, "recall@1": recall_1/n, "recall@5": recall_5/n,
         "recall@10": recall_10/n, "mrr": mrr/n}
    print(f"  α={alpha}  R@1={r['recall@1']:.4f}  R@5={r['recall@5']:.4f}"
          f"  R@10={r['recall@10']:.4f}  MRR={r['mrr']:.4f}")
    return r

dev_global_idx  = [queries_df.index[queries_df["split"]=="dev"][i]
                   for i in range(len(dev_df))]
test_global_idx = [queries_df.index[queries_df["split"]=="test"][i]
                   for i in range(len(test_df))]

alphas = [0.9, 0.7, 0.5, 0.3]

print("\n── DEV SPLIT — Alpha sweep ──────────────────────")
dev_all = []
for alpha in alphas:
    dev_all.append(evaluate_combined(dev_df, dev_global_idx, alpha, "Dev"))

best_alpha = max(dev_all, key=lambda x: x["mrr"])["alpha"]
print(f"\nBest α on dev (by MRR): {best_alpha}")

print("\n── TEST SPLIT — Best alpha ──────────────────────")
test_result = evaluate_combined(test_df, test_global_idx, best_alpha, "Test")

with open("il_pcsr/bm25_results.json") as f:
    bm25_results = json.load(f)

bm25_test = bm25_results["test"]
print("\n── FINAL RESULTS TABLE ──────────────────────────")
print(f"{'System':<30} {'R@1':>6} {'R@5':>6} {'R@10':>7} {'MRR':>6}")
print("-" * 58)
print(f"{'BM25 (baseline)':<30}"
      f" {bm25_test['recall@1']:>6.4f}"
      f" {bm25_test['recall@5']:>6.4f}"
      f" {bm25_test['recall@10']:>7.4f}"
      f" {bm25_test['mrr']:>6.4f}")
print(f"{'BM25 + Jaccard (α='+str(best_alpha)+')':<30}"
      f" {test_result['recall@1']:>6.4f}"
      f" {test_result['recall@5']:>6.4f}"
      f" {test_result['recall@10']:>7.4f}"
      f" {test_result['mrr']:>6.4f}")

final = {
    "bm25_test": bm25_test,
    "jaccard_dev_sweep": dev_all,
    "best_alpha": best_alpha,
    "jaccard_test": test_result
}
with open("results/maps/final_results.json", "w") as f:
    json.dump(final, f, indent=2)
print("\nSaved final_results.json")
