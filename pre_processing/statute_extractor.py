import pandas as pd
import numpy as np
import json, re
import ahocorasick
from collections import defaultdict

# pip install pyahocorasick  ← run this first if not installed

precedents_df = pd.read_json("il_pcsr/precedents.json", lines=True)
statutes_df   = pd.read_json("il_pcsr/statutes.json",   lines=True)

def to_list(val):
    if val is None: return []
    if isinstance(val, (list, np.ndarray)): return list(val)
    return []

ACT_ABBREVS = {
    "the indian penal code, 1860"             : ["ipc", "i.p.c", "i.p.c."],
    "the code of criminal procedure, 1973"    : ["crpc", "cr.pc", "cr.p.c", "cr.p.c."],
    "the code of civil procedure, 1908"       : ["cpc", "c.p.c", "c.p.c."],
    "the constitution of india, 1949"         : ["constitution of india", "the constitution"],
    "the negotiable instruments act, 1881"    : ["ni act", "n.i. act"],
    "the motor vehicles act, 1988"            : ["mv act", "m.v. act"],
    "the income tax act, 1961"                : ["it act", "i.t. act"],
    "the protection of children from sexual offences act, 2012": ["pocso"],
    "the prevention of corruption act, 1988"  : ["pc act", "p.c. act"],
    "the arbitration and conciliation act, 1996": ["arbitration act"],
}

def build_variants(provision_name):
    raw  = provision_name.strip()
    low  = raw.lower()
    variants = set()
    variants.add(low)
    m = re.match(
        r'^(section|article|rule|order|schedule)\s+([\w\-]+(?:\([^\)]*\))*)\s+(?:in|of)\s+(.+)$',
        low, re.IGNORECASE
    )
    if m:
        ptype, num, act = m.group(1), m.group(2), m.group(3).strip().rstrip(',')
        for abbr_act, abbr_list in ACT_ABBREVS.items():
            if abbr_act in act.lower():
                for abbr in abbr_list:
                    variants.add(f"{ptype} {num} {abbr}")
                    variants.add(f"s. {num} {abbr}")
                    variants.add(f"sec. {num} {abbr}")
                    variants.add(f"sec {num} {abbr}")
                    variants.add(f"s {num} {abbr}")
        act_no_year = re.sub(r',?\s*\d{4}$', '', act).strip()
        variants.add(f"{ptype} {num} of {act_no_year}")
        variants.add(f"{ptype} {num} of the {act_no_year}")
        variants.add(f"{ptype} {num} of {act}")
    return variants

# ── Build Aho-Corasick automaton ──────────────────────────────────────────
print("Building automaton...")
A = ahocorasick.Automaton()
for _, row in statutes_df.iterrows():
    sid      = str(row["id"])
    variants = build_variants(str(row["provision_name"]))
    for v in variants:
        # store (sid) — if key already exists keep both
        if v in A:
            A.get(v).add(sid)
        else:
            A.add_word(v, {sid})

A.make_automaton()
print(f"Automaton built — {len(A)} patterns indexed")

def extract_statutes_fast(text):
    if not isinstance(text, str): return set()
    text_low = text.lower()
    matched  = set()
    for _, sid_set in A.iter(text_low):
        matched.update(sid_set)
    return matched

# ── Validate on labeled precedents ───────────────────────────────────────
labeled = precedents_df[
    precedents_df["relevant_statute_ids"].apply(lambda x: len(to_list(x)) > 0)
].copy()

print(f"Validating on {len(labeled)} labeled precedents...")
tp_total = fn_total = fp_total = zero_recovery = 0

for i, (_, row) in enumerate(labeled.iterrows()):
    if i % 200 == 0:
        print(f"  {i}/{len(labeled)}...")
    label_ids = set(str(s) for s in to_list(row["relevant_statute_ids"]))
    extracted = extract_statutes_fast(str(row["text"]))
    tp = len(label_ids & extracted)
    fn = len(label_ids - extracted)
    fp = len(extracted - label_ids)
    tp_total += tp
    fn_total += fn
    fp_total += fp
    if tp == 0:
        zero_recovery += 1

precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0
recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0
f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0

print(f"\n── VALIDATION RESULTS ───────────────────────────")
print(f"  Precision  : {precision:.3f}")
print(f"  Recall     : {recall:.3f}")
print(f"  F1         : {f1:.3f}")
print(f"  Zero recovery: {zero_recovery}/{len(labeled)} ({zero_recovery/len(labeled)*100:.1f}%)")

# ── Extract from ALL precedents ───────────────────────────────────────────
print(f"\nExtracting from all {len(precedents_df)} precedents...")
precedent_statute_map = {}

for i, (_, row) in enumerate(precedents_df.iterrows()):
    if i % 500 == 0:
        print(f"  {i}/{len(precedents_df)}...")
    pid       = str(row["id"])
    extracted = extract_statutes_fast(str(row["text"]))
    labeled   = set(str(s) for s in to_list(row["relevant_statute_ids"]))
    precedent_statute_map[pid] = list(labeled | extracted)

counts = pd.Series([len(v) for v in precedent_statute_map.values()])
print(f"\n── FINAL MAP ─────────────────────────────────────")
print(f"  Still zero : {(counts==0).sum()}")
print(f"  Mean       : {counts.mean():.2f}")
print(f"  Median     : {counts.median():.1f}")

with open("results/primary_analysis/precedent_statute_map.json", "w") as f:
    json.dump(precedent_statute_map, f)
print("Saved to results/primary_analysis/precedent_statute_map.json")