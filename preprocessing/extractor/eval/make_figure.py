"""Robustness-curve figure for the poster: F1 vs degradation severity."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "cache_v2/full_frame_sweep.csv")

series = [("w3robust_F1", "Student-w3 robust (ours)", "C2", "-o"),
          ("w3clean_F1", "Student-w3 clean", "C0", "--s"),
          ("mediapipe_F1", "MediaPipe (native)", "C3", "-^")]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, cond in zip(axes, ["low_light", "motion_blur"]):
    sub = df[(df.condition == cond) | (df.condition == "clean")].copy()
    sub = pd.concat([df[df.condition == "clean"].assign(severity=0.0), df[df.condition == cond]])
    sub = sub.sort_values("severity")
    for col, label, color, style in series:
        ax.plot(sub.severity, sub[col], style, color=color, label=label, linewidth=2, markersize=6)
    ax.set_title(cond.replace("_", " ").title())
    ax.set_xlabel("degradation severity")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("macro-F1 (through frozen classifier)")
axes[0].legend(loc="lower left", fontsize=9)
fig.suptitle("Robustness: landmark extractor F1 vs adverse-condition severity", fontweight="bold")
fig.tight_layout()
out = ROOT / "cache_v2/robustness_curves.png"
fig.savefig(out, dpi=130)
print(f"[done] -> {out}")
