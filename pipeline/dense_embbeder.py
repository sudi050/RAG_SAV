from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
import os

# ── Load model ────────────────────────────────────────────────────────────
model = SentenceTransformer("law-ai/InLegalBERT")
model = model.to("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
print(f"Model loaded: {model}")
print(f"Model device: {model.device}")

# ── Load data ─────────────────────────────────────────────────────────────
precedents_df = pd.read_json("./il_pcsr/precedents.json", lines=True)
train_df      = pd.read_json("./il_pcsr/train_queries.json", lines=True)
dev_df        = pd.read_json("./il_pcsr/dev_queries.json",   lines=True)
test_df       = pd.read_json("./il_pcsr/test_queries.json",  lines=True)
queries_df    = pd.concat([train_df, dev_df, test_df], ignore_index=True)

def safe_text(t):
    if isinstance(t, list):
        t = " ".join(str(x) for x in t)
    if not isinstance(t, str):
        t = str(t) if t else " "
    return t.strip() if t.strip() else " "

prec_texts  = [safe_text(t) for t in precedents_df["text"].fillna("").tolist()]
query_texts = [safe_text(t) for t in queries_df["text"].fillna("").tolist()]


print(f"Precedents : {len(prec_texts)}")
print(f"Queries    : {len(query_texts)}")

# ── Encode precedents ─────────────────────────────────────────────────────
print("\nEncoding precedents...")
prec_embs = model.encode(
    prec_texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
np.save("./results/maps/prec_embeddings.npy", prec_embs)
print(f"Precedent embeddings saved: {prec_embs.shape}")

# ── Encode queries ────────────────────────────────────────────────────────
print("\nEncoding queries...")
query_embs = model.encode(
    query_texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
np.save("./results/maps/query_embeddings.npy", query_embs)
print(f"Query embeddings saved: {query_embs.shape}")

# ── Compute cosine scores (dot product, already normalized) ───────────────
print("\nComputing cosine scores...")
dense_scores = (query_embs @ prec_embs.T).tolist()   # shape: 6271 x 3183

with open("./results/maps/dense_scores.json", "w") as f:
    json.dump(dense_scores, f)
print("dense_scores.json saved.")
