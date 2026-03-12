import pandas as pd

# ── Load ──────────────────────────────────────────────────────────────────
train_df = pd.read_json("il_pcsr/train_queries.json", lines=True)
dev_df   = pd.read_json("il_pcsr/dev_queries.json",   lines=True)
test_df  = pd.read_json("il_pcsr/test_queries.json",  lines=True)

train_df["split"] = "train"
dev_df["split"]   = "dev"
test_df["split"]  = "test"

queries_df    = pd.concat([train_df, dev_df, test_df], ignore_index=True)
precedents_df = pd.read_json("il_pcsr/precedents.json", lines=True)

print(f"Total queries   : {len(queries_df)}")
print(f"Total precedents: {len(precedents_df)}")

# ── 1. Overlap by ID ──────────────────────────────────────────────────────
q_ids = set(queries_df["id"].dropna().astype(str))
p_ids = set(precedents_df["id"].dropna().astype(str))

overlap_ids = q_ids & p_ids
print(f"\n── BY ID ────────────────────────────────────────")
print(f"  Overlapping IDs     : {len(overlap_ids)}")
print(f"  Only in queries     : {len(q_ids - p_ids)}")
print(f"  Only in precedents  : {len(p_ids - q_ids)}")

if overlap_ids:
    overlap_df = queries_df[queries_df["id"].astype(str).isin(overlap_ids)][
        ["id", "case_title", "date", "split"]
    ]
    print(f"\n  Overlapping cases:")
    print(overlap_df.to_string(index=False))

# ── 2. Overlap by case_title ──────────────────────────────────────────────
q_titles = set(queries_df["case_title"].dropna().str.strip().str.lower())
p_titles = set(precedents_df["case_title"].dropna().str.strip().str.lower())

overlap_titles = q_titles & p_titles
print(f"\n── BY CASE TITLE ────────────────────────────────")
print(f"  Overlapping titles  : {len(overlap_titles)}")
print(f"  Only in queries     : {len(q_titles - p_titles)}")
print(f"  Only in precedents  : {len(p_titles - q_titles)}")

if overlap_titles:
    matched = queries_df[
        queries_df["case_title"].str.strip().str.lower().isin(overlap_titles)
    ][["id", "case_title", "date", "split"]]
    print(f"\n  Sample overlapping titles (top 20):")
    print(matched.head(20).to_string(index=False))

# ── 3. Overlap by text hash (exact duplicate detection) ───────────────────
# print(f"\n── BY TEXT (exact match via hash) ───────────────")
# queries_df["text_hash"]    = queries_df["text"].apply(
#     lambda x: hash(str(x).strip()) if pd.notna(x) else None
# )
# precedents_df["text_hash"] = precedents_df["text"].apply(
#     lambda x: hash(str(x).strip()) if pd.notna(x) else None
# )

# q_hashes = set(queries_df["text_hash"].dropna())
# p_hashes = set(precedents_df["text_hash"].dropna())

# overlap_hashes = q_hashes & p_hashes
# print(f"  Exact text duplicates: {len(overlap_hashes)}")

# ── 4. Cross-reference: queries that cite a precedent that IS also a query ─
# print(f"\n── CROSS-REFERENCE: queries cited as precedents ─")
# cited_ids = set()
# for val in queries_df["relevant_precedent_ids"].dropna():
#     cited_ids.update(to_str_list(val))

# cited_and_also_query = cited_ids & q_ids
# print(f"  Unique precedent IDs cited by queries  : {len(cited_ids)}")
# print(f"  Of those, also present as a query case : {len(cited_and_also_query)}")
# if cited_and_also_query:
#     sample = queries_df[queries_df["id"].astype(str).isin(cited_and_also_query)][
#         ["id", "case_title", "date", "split"]
#     ]
#     print(f"\n  Sample (top 10):")
#     print(sample.head(10).to_string(index=False))

# ── 5. Summary ────────────────────────────────────────────────────────────
print(f"\n── SUMMARY ──────────────────────────────────────")
print(f"  ID overlap          : {len(overlap_ids)}")
print(f"  Title overlap       : {len(overlap_titles)}")
# print(f"  Exact text overlap  : {len(overlap_hashes)}")
