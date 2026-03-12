import pandas as pd
import json, os, re

train_df = pd.read_json("il_pcsr/train_queries.json", lines=True)
dev_df   = pd.read_json("il_pcsr/dev_queries.json",   lines=True)
test_df  = pd.read_json("il_pcsr/test_queries.json",  lines=True)

precedents_df = pd.read_json("il_pcsr/precedents.json", lines=True)
statutes_df   = pd.read_json("il_pcsr/statutes.json",   lines=True)


train_df["split"] = "train"
dev_df["split"]   = "dev"
test_df["split"]  = "test"

queries_df    = pd.concat([train_df, dev_df, test_df], ignore_index=True)

print("── COLUMNS ─────────────────────────────────────")
print("Queries   :", queries_df.columns.tolist())
print("Precedents:", precedents_df.columns.tolist())
print("Statutes  :", statutes_df.columns.tolist())

def extract_year(text):
    if pd.isna(text):
        return None
    matches = re.findall(r'\b(1[89]\d{2}|20[012]\d)\b', str(text))
    return int(matches[0]) if matches else None

date_col_candidates = ["date", "year", "judgment_date", "decided_on", "doc_id", "id"]

def attach_years(df, label):
    for col in date_col_candidates:
        if col in df.columns:
            df["year"] = df[col].apply(extract_year)
            if df["year"].notna().sum() > 10:
                print(f"\n[{label}] Year extracted from: '{col}'")
                return df
    # Fallback: scan first 300 chars of full text
    text_col = next((c for c in ["full_text","text","judgment","content"] if c in df.columns), None)
    if text_col:
        df["year"] = df[text_col].apply(lambda x: extract_year(str(x)[:300]))
        print(f"\n[{label}] Year extracted from text column: '{text_col}'")
    return df

queries_df    = attach_years(queries_df,    "QUERIES")
precedents_df = attach_years(precedents_df, "PRECEDENTS")

for label, df in [("QUERIES", queries_df), ("PRECEDENTS", precedents_df)]:
    yr = df["year"].dropna()
    print(f"\n── {label} DATE RANGE ──────────────────────")
    print(f"  Total cases  : {len(df)}")
    print(f"  Earliest year: {int(yr.min()) if len(yr) else 'N/A'}")
    print(f"  Latest year  : {int(yr.max()) if len(yr) else 'N/A'}")
    print(f"  No date found: {df['year'].isna().sum()}")
    print(f"\n  Year distribution:")
    print(yr.value_counts().sort_index().to_string())

print("\n── STATUTE ANNOTATION COVERAGE ─────────────────")
stat_col = next(
    (c for c in ["relevant_statutes","statute_ids","statutes","relevant_statute_ids","positive_statutes"]
     if c in queries_df.columns), None
)
if stat_col:
    no_statute = queries_df[stat_col].apply(
        lambda x: x is None or (isinstance(x, list) and len(x) == 0)
    )
    print(f"  Statute column        : '{stat_col}'")
    print(f"  Cases WITH statutes   : {(~no_statute).sum()}")
    print(f"  Cases WITHOUT statutes: {no_statute.sum()}")
    print(f"  Coverage %            : {(~no_statute).mean()*100:.1f}%")
    print(f"\n  Per split:")
    for split in ["train", "dev", "test"]:
        s = queries_df[queries_df["split"] == split][stat_col]
        has = s.apply(lambda x: isinstance(x, list) and len(x) > 0).sum()
        print(f"    {split:<6}: {has}/{len(s)} with statutes ({has/len(s)*100:.1f}%)")
else:
    print("  ⚠ No statute column found. Columns:", queries_df.columns.tolist())

print("\n── STATUTE ANNOTATION COVERAGE (PRECEDENTS) ─────")
p_stat_col = next(
    (c for c in ["relevant_statutes", "relevant_statute_ids"]
     if c in precedents_df.columns), None
)
if p_stat_col:
    no_stat_p = precedents_df[p_stat_col].apply(
        lambda x: x is None or (isinstance(x, list) and len(x) == 0)
    )
    print(f"  Statute column          : '{p_stat_col}'")
    print(f"  Precedents WITH statutes: {(~no_stat_p).sum()}")
    print(f"  Precedents WITHOUT      : {no_stat_p.sum()}")
    print(f"  Coverage %              : {(~no_stat_p).mean()*100:.1f}%")

print("\n── RHETORICAL ROLE ANNOTATION ───────────────────")
rhet_col = next(
    (c for c in ["rhetorical_roles","roles","role_labels","annotations"]
     if c in queries_df.columns), None
)
if rhet_col:
    no_rhet = queries_df[rhet_col].apply(
        lambda x: x is None or (isinstance(x, list) and len(x) == 0)
    )
    print(f"  Rhetorical role column: '{rhet_col}'")
    print(f"  Cases WITH roles      : {(~no_rhet).sum()}")
    print(f"  Cases WITHOUT roles   : {no_rhet.sum()}")
else:
    print("  ⚠ No rhetorical role column — IL-PCSR does not include this annotation.")
    print("    (Use LegalSeg dataset separately for rhetorical role labels)")

print("\n── RHETORICAL ROLE ANNOTATION (PRECEDENTS) ─────────────────────────────")
p_rhet_col = next(
    (c for c in ["rhetorical_roles","roles","role_labels","annotations"]
     if c in precedents_df.columns), None
)
if p_rhet_col:
    no_rhet_p = precedents_df[p_rhet_col].apply(
        lambda x: x is None or (isinstance(x, list) and len(x) == 0)
    )
    print(f"  Rhetorical role column: '{p_rhet_col}'")
    print(f"  Precedents WITH roles : {(~no_rhet_p).sum()}")
    print(f"  Precedents WITHOUT    : {no_rhet_p.sum()}")
else:
    print("  ⚠ No rhetorical role column in precedents.")

summary = pd.DataFrame([
    {
        "config"    : "queries",
        "total"     : len(queries_df),
        "year_min"  : int(queries_df["year"].min()) if queries_df["year"].notna().any() else None,
        "year_max"  : int(queries_df["year"].max()) if queries_df["year"].notna().any() else None,
        "no_date"   : int(queries_df["year"].isna().sum()),
        "no_statute": int(no_statute.sum()) if stat_col else "N/A",
        "no_rhetorical_role": int(no_rhet.sum()) if rhet_col else "N/A",
    },
    {
        "config"    : "precedents",
        "total"     : len(precedents_df),
        "year_min"  : int(precedents_df["year"].min()) if precedents_df["year"].notna().any() else None,
        "year_max"  : int(precedents_df["year"].max()) if precedents_df["year"].notna().any() else None,
        "no_date"   : int(precedents_df["year"].isna().sum()),
        "no_statute": int(no_stat_p.sum()) if p_stat_col else "N/A",
        "no_rhetorical_role": int(no_rhet_p.sum()) if p_rhet_col else "N/A",
    },
])
summary.to_csv("il_pcsr/coverage_summary.csv", index=False)
print("\nSummary saved to il_pcsr/coverage_summary.csv")
