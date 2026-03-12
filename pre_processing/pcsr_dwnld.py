from datasets import load_dataset
import os

HF_TOKEN = os.getenv("HF_TOKEN")

os.makedirs("il_pcsr", exist_ok=True)

print("Loading queries...")
ds_queries = load_dataset("Exploration-Lab/IL-PCSR", name="queries", token=HF_TOKEN)

print("Loading precedents...")
ds_precedents = load_dataset("Exploration-Lab/IL-PCSR", name="precedents", token=HF_TOKEN)

print("Loading statutes...")
ds_statutes = load_dataset("Exploration-Lab/IL-PCSR", name="statutes", token=HF_TOKEN)

for split in ["train_queries", "dev_queries", "test_queries"]:
    ds_queries[split].to_json(
        f"il_pcsr/{split}.json",
        force_ascii=False
    )
    print(f"Saved {split} — {len(ds_queries[split])} records")

ds_precedents["precedent_candidates"].to_json(
    "il_pcsr/precedents.json",
    force_ascii=False
)
print(f"Saved precedents — {len(ds_precedents['precedent_candidates'])} records")

# Save statutes
ds_statutes["statute_candidates"].to_json(
    "il_pcsr/statutes.json",
    force_ascii=False
)
print(f"Saved statutes — {len(ds_statutes['statute_candidates'])} records")

print("\n── Saved files ──────────────────────────")
for f in os.listdir("il_pcsr"):
    size_kb = os.path.getsize(f"il_pcsr/{f}") / 1024
    print(f"  {f:<30} {size_kb:>8.1f} KB")
