import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, re

train_df = pd.read_json("il_pcsr/train_queries.json", lines=True)
dev_df   = pd.read_json("il_pcsr/dev_queries.json",   lines=True)
test_df  = pd.read_json("il_pcsr/test_queries.json",  lines=True)

train_df["split"] = "train"
dev_df["split"]   = "dev"
test_df["split"]  = "test"

queries_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

def extract_year(val):
    if pd.isna(val): return None
    m = re.findall(r'\b(1[89]\d{2}|20[012]\d)\b', str(val))
    return int(m[0]) if m else None

queries_df["year"] = queries_df["date"].apply(extract_year)

print("── DATE RANGE PER SPLIT ─────────────────────────")
for split in ["train", "dev", "test"]:
    df = queries_df[queries_df["split"] == split]
    yr = df["year"].dropna()
    print(f"\n  {split.upper()} ({len(df)} cases, {df['year'].isna().sum()} undated)")
    print(f"    Range : {int(yr.min())} – {int(yr.max())}")
    print(f"    Median: {int(yr.median())}")

splits      = ["train", "dev", "test"]
split_data  = {}
all_years   = set()

for split in splits:
    df  = queries_df[queries_df["split"] == split]
    vc  = df["year"].dropna().astype(int).value_counts().sort_index()
    split_data[split] = vc
    all_years.update(vc.index.tolist())

all_years = sorted(all_years)

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Cases per Year by Split", "Cumulative Cases Over Time"),
    vertical_spacing=0.14
)

colors = {"train": None, "dev": None, "test": None}

# Panel 1 — stacked bar per split
for split in splits:
    vc = split_data[split]
    fig.add_trace(go.Bar(
        x=[str(y) for y in all_years],
        y=[vc.get(y, 0) for y in all_years],
        name=split.capitalize(),
    ), row=1, col=1)

# Panel 2 — cumulative line per split
for split in splits:
    vc     = split_data[split]
    counts = [vc.get(y, 0) for y in all_years]
    cumsum = list(pd.Series(counts).cumsum())
    fig.add_trace(go.Scatter(
        x=[str(y) for y in all_years],
        y=cumsum,
        name=f"{split.capitalize()} (cum.)",
        mode="lines",
    ), row=2, col=1)

fig.update_layout(
    title={"text": "Queries Skew Post-2010; Splits Share Same Range<br><span style='font-size:16px;font-weight:normal;'>Source: IL-PCSR | 6,271 query cases across train/dev/test</span>"},
    barmode="stack",
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)
fig.update_xaxes(title_text="Year", tickangle=45, nticks=20)
fig.update_yaxes(title_text="Cases")

fig.write_image("results/primary_analysis/plots/queries_split_dist.png")
with open("results/primary_analysis/queries_split_dist.png.meta.json", "w") as f:
    json.dump({
        "caption": "IL-PCSR Queries: Year Distribution by Split (1951–2023)",
        "description": "Two-panel chart: stacked bar of cases per year per split, and cumulative case growth per split"
    }, f)

print("\nSaved queries_split_dist.png")
