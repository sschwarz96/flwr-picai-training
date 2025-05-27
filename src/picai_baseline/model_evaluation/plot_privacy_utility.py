#!/usr/bin/env python3
"""
Plot AUROC vs ε when your JSONs sit directly under
epsilonX/ (no timestamp) or under epsilonX/<timestamp>/
"""
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Gather all inference_results.json files
RESULTS_DIR = Path('/home/zimon/flwr-picai-training/outputs/final_results/no_DA')

print(f"🔍 Searching in: {RESULTS_DIR}")
print(f"Directory exists: {RESULTS_DIR.exists()}")

if not RESULTS_DIR.exists():
    sys.exit(f"❌ Results directory does not exist: {RESULTS_DIR}")

records = []
found_files = []

for jpath in RESULTS_DIR.rglob("inference_results.json"):
    found_files.append(str(jpath))
    print(f"📁 Found file: {jpath.relative_to(RESULTS_DIR)}")

    # walk up until we find an epsilonX or no_DP folder
    eps = None
    for p in jpath.parents:
        print(f"  Checking parent: {p.name}")
        if m := re.match(r"epsilon(\d+)", p.name):
            eps = int(m.group(1))
            print(f"  ✅ Found epsilon: {eps}")
            break
        if p.name.startswith("no_DP"):
            eps = np.inf
            print(f"  ✅ Found no_DP case")
            break
        if p == RESULTS_DIR:
            print(f"  ⚠️  Reached root directory without finding epsilon")
            break

    if eps is None:
        print(f"  ❌ Could not determine epsilon for {jpath}")
        continue

    try:
        with open(jpath, 'r') as f:
            data = json.load(f)

        print(f"  📊 JSON keys: {list(data.keys())}")

        # Try multiple possible key names for AUROC
        auroc = None
        possible_keys = ["auroc", "AUROC", "auc", "AUC", "roc_auc", "ROC_AUC"]

        for key in possible_keys:
            if key in data:
                auroc = data[key]
                print(f"  ✅ Found AUROC with key '{key}': {auroc}")
                break

        if auroc is None:
            print(f"  ❌ No AUROC found in {jpath}")
            print(f"  Available keys: {list(data.keys())}")
            continue

        # Handle case where auroc might be a list or nested structure
        if isinstance(auroc, (list, tuple)):
            if len(auroc) > 0:
                auroc = auroc[0] if isinstance(auroc[0], (int, float)) else float(auroc[0])
            else:
                print(f"  ❌ AUROC list is empty")
                continue
        elif isinstance(auroc, dict):
            # If it's a dict, try to find a reasonable value
            if 'value' in auroc:
                auroc = auroc['value']
            elif 'mean' in auroc:
                auroc = auroc['mean']
            else:
                print(f"  ❌ AUROC is dict but no 'value' or 'mean' key: {auroc}")
                continue

        # Ensure auroc is a number
        try:
            auroc = float(auroc)
        except (ValueError, TypeError):
            print(f"  ❌ Could not convert AUROC to float: {auroc}")
            continue

        print(f"  ✅ Using AUROC: {auroc}")
        records.append({"epsilon": eps, "auroc": auroc})

    except Exception as e:
        print(f"  ❌ Error reading {jpath}: {e}")
        continue

print(f"\n📈 Summary:")
print(f"Found {len(found_files)} inference_results.json files")
print(f"Successfully processed {len(records)} files")

if not records:
    print("\n❌ No valid records found!")
    print("Possible issues:")
    print("1. No inference_results.json files found")
    print("2. Files don't contain 'auroc' key")
    print("3. Directory structure doesn't match expected pattern")
    sys.exit(1)

# Print all records for debugging
print(f"\n📋 All records:")
for r in records:
    print(f"  ε={r['epsilon']}, AUROC={r['auroc']}")

df = pd.DataFrame(records)

# 2) Compute mean/std per ε
grp = df.groupby("epsilon").auroc.agg(["mean", "std", "count"]).reset_index().sort_values("epsilon")
grp["std"] = grp["std"].fillna(0.0)

print(f"\n📊 Grouped results:")
for _, row in grp.iterrows():
    eps_str = f"ε={row['epsilon']}" if np.isfinite(row['epsilon']) else "no-DP"
    print(f"  {eps_str}: mean={row['mean']:.3f}, std={row['std']:.3f}, count={row['count']}")

# 3) Build linear x positions and labels
eps_vals = grp["epsilon"].tolist()
labels = [str(int(e)) if np.isfinite(e) else "no-DP" for e in eps_vals]
xpos = list(range(len(labels)))

# 4) Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot with error bars
ax.errorbar(
    xpos, grp["mean"], yerr=grp["std"],
    fmt="o-", capsize=4, linewidth=2, markersize=8
)

ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_xlabel("Privacy budget ε", fontsize=12)
ax.set_ylabel("AUROC", fontsize=12)
ax.set_ylim(0.0, 1.0)
ax.grid(True, ls="--", lw=0.4, alpha=0.7)

# Add title
ax.set_title("AUROC vs Privacy Budget (ε)", fontsize=14, fontweight='bold')

# Annotate points with values
for xi, (m, s) in enumerate(zip(grp["mean"], grp["std"])):
    # Show mean value above point
    ax.annotate(f"{m:.3f}",
                (xi, m),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=10, fontweight='bold')

plt.tight_layout()

# Save the plot
out = RESULTS_DIR / "AUROC_vs_epsilon_linear_no_DA.png"
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved plot → {out}")

# Also show the plot if running interactively
plt.show()