#!/usr/bin/env python3
# save as plot_privacy_utility.py and run: python plot_privacy_utility.py

#!/usr/bin/env python3
"""
Plot Average Precision (AP) vs ε when your JSONs sit directly under
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
RESULTS_DIR = Path('/home/zimon/flwr-picai-training/outputs/final_results')
records = []

for jpath in RESULTS_DIR.rglob("inference_results.json"):
    # walk up until we find an epsilonX or no_DP folder
    eps = None
    for p in jpath.parents:
        if m := re.match(r"epsilon(\d+)_DA", p.name):
            eps = int(m.group(1))
            break
        if p.name.startswith("no_DP"):
            eps = np.inf
            break
        if p == RESULTS_DIR:
            break
    if eps is None:
        continue

    data = json.loads(jpath.read_text())
    ap   = data.get("average_precision")
    if ap is None:
        continue

    records.append({"epsilon": eps, "AP": ap})

if not records:
    sys.exit("❌ No runs found!")

df = pd.DataFrame(records)

# 2) Compute mean/std per ε
grp = df.groupby("epsilon").AP.agg(["mean", "std"]).reset_index().sort_values("epsilon")
grp["std"] = grp["std"].fillna(0.0)

# 3) Build linear x positions and labels
eps_vals = grp["epsilon"].tolist()
labels   = [str(int(e)) if np.isfinite(e) else "no-DP" for e in eps_vals]
xpos     = list(range(len(labels)))

# 4) Plot
fig, ax = plt.subplots(figsize=(6,4))
ax.errorbar(
    xpos, grp["mean"], yerr=grp["std"],
    fmt="o-", capsize=4
)

ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_xlabel("Privacy budget ε")
ax.set_ylabel("Average Precision")
ax.set_yscale("log")
ax.set_ylim(1e-5, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0e}"))
ax.grid(True, ls="--", lw=0.4)

# annotate
for xi, (m, s) in enumerate(zip(grp["mean"], grp["std"])):
    ax.annotate(f"{m:.2e}",
                (xi, m),
                textcoords="offset points", xytext=(0,5),
                ha="center", fontsize=8)

plt.tight_layout()
out = RESULTS_DIR / "AP_vs_epsilon_linear_DA.png"
fig.savefig(out, dpi=300)
print(f"✅ Saved → {out}")