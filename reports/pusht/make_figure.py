"""Render the headline figure from evidence.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
data = json.loads((ROOT / "evidence.json").read_text())["observed"]
names = ["MSE", "Drift", "Diffusion\n(20-step)"]
keys = ["mse", "drift", "diffusion"]
colors = ["#2878B5", "#D9534F", "#6C757D"]

panels = [
    ("heldout_mse", "Held-out one-pass MSE", True),
    ("rollout_mse", "64-frame rollout MSE", True),
    ("latency_ms_per_frame", "Latency (ms / frame)", True),
]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
for ax, (prefix, title, log_scale) in zip(axes, panels):
    means = [data[k][f"{prefix}_mean"] for k in keys]
    stds = [data[k][f"{prefix}_std"] for k in keys]
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.92)
    ax.set_xticks(x, names)
    ax.set_title(title, fontsize=11, weight="bold")
    if log_scale:
        ax.set_yscale("log")
    for bar, value in zip(bars, means):
        label = f"{value:.4g}" if value < 1 else f"{value:.2f}"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

fig.suptitle("Observed Push-T: one-pass speed aligns; reconstructed drift quality does not",
             fontsize=13, weight="bold")
fig.text(0.5, 0.01, "Mean ± population SD across 8 independently trained seeds; lower is better",
         ha="center", fontsize=9, color="#444444")
fig.tight_layout(rect=(0, 0.05, 1, 0.92))
out = ROOT / "images" / "quality_latency.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(out)
