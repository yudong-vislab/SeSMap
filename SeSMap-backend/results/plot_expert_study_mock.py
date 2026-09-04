"""Create an illustrative expert-study figure for layout validation.

All participant-level values in this script are simulated and must be replaced
with verified study logs before the figure is used in a manuscript.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent

participants = ["E1", "E2", "E3", "E4", "E5", "E6"]
metrics = {
    "Task time\n(min)": np.array([21, 17, 24, 26, 29, 32]),
    "Unique HSUs\ninspected": np.array([18, 22, 25, 29, 34, 41]),
    "MSUs retained\nin synthesis": np.array([9, 10, 10, 11, 12, 12]),
    "Source verification\ncoverage (%)": np.array([88.9, 90.0, 90.0, 90.9, 100.0, 100.0]),
}

likert = np.array([
    [0, 0, 0, 0, 6],  # Q1 map interpretability
    [0, 0, 0, 1, 5],  # Q2 related-statement discovery
    [0, 0, 0, 1, 5],  # Q3 source verification
    [0, 0, 0, 0, 6],  # Q4 synthesis organization
    [0, 0, 1, 1, 4],  # Q5 manageable analytical effort
])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

fig = plt.figure(figsize=(3.45, 5.9), constrained_layout=False)
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.35], hspace=0.72, wspace=0.56)

point_color = "#2563A6"
box_color = "#BDD7EE"
rng = np.random.default_rng(7)

for idx, (label, values) in enumerate(metrics.items()):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.boxplot(
        values,
        positions=[1],
        widths=0.42,
        patch_artist=True,
        whis=(0, 100),
        showfliers=False,
        boxprops={"facecolor": box_color, "edgecolor": "#4A6073", "linewidth": 0.9},
        medianprops={"color": "#9C2F2F", "linewidth": 1.4},
        whiskerprops={"color": "#4A6073", "linewidth": 0.9},
        capprops={"color": "#4A6073", "linewidth": 0.9},
    )
    jitter = rng.uniform(-0.08, 0.08, len(values))
    ax.scatter(np.ones(len(values)) + jitter, values, s=20, color=point_color,
               edgecolor="white", linewidth=0.45, zorder=3)
    median = np.median(values)
    ax.set_title(f"{label}\nMdn = {median:g}", pad=3)
    ax.set_xticks([])
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.65)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
fig.text(0.04, 0.965, "(a)", fontsize=8.5, fontweight="bold", va="top")

ax = fig.add_subplot(gs[2, :])
labels = ["Q1 Map", "Q2 Discovery", "Q3 Verification",
          "Q4 Synthesis", "Q5 Effort"]
colors = ["#B2182B", "#EF8A62", "#D9D9D9", "#67A9CF", "#2166AC"]
left = np.zeros(5)
for response_idx in range(5):
    vals = likert[:, response_idx]
    bars = ax.barh(np.arange(5), vals, left=left, height=0.58,
                   color=colors[response_idx], edgecolor="white", linewidth=0.6,
                   label=str(response_idx + 1))
    for bar, value in zip(bars, vals):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(value)), ha="center", va="center",
                    fontsize=7, color="white" if response_idx in (0, 1, 3, 4) else "#333333")
    left += vals

ax.set_yticks(np.arange(5), labels)
ax.invert_yaxis()
ax.set_xlim(0, 6)
ax.set_xlabel("Participants")
ax.set_title("(b) Five-point Likert responses", loc="left",
             fontsize=8.5, fontweight="bold", pad=6)
ax.grid(axis="x", color="#E6E8EB", linewidth=0.65)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.legend(title="Rating", ncol=5, frameon=False, loc="upper center",
          bbox_to_anchor=(0.5, -0.34), columnspacing=0.7, handlelength=1.0,
          handletextpad=0.35)

fig.subplots_adjust(left=0.20, right=0.97, top=0.94, bottom=0.11)

for suffix in ("pdf", "png"):
    fig.savefig(OUT / f"expert_study_results_mock.{suffix}", dpi=300,
                bbox_inches="tight", facecolor="white")

print("Simulated participant data")
for label, values in metrics.items():
    print(label.replace("\n", " "), values.tolist(), "median=", float(np.median(values)),
          "range=", (float(values.min()), float(values.max())))
