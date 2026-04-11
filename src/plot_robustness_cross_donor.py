"""
Robustness plot — same format as plot_robustness.py but including
cross-donor results (8K Donor B, 3K Donor C) alongside the 68K splits.
Each model×representation gets: 68K split range + mean dot, 8K dot, 3K dot.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Paths ──
PCA_SPLITS = "results/full_train_all_pca/reports/robustness_splits_full.csv"
HVG_SPLITS = "results/full_train_all_hvg/reports/robustness_splits_full.csv"
EXT_8K = "results/external_validation/external_validation_summary.csv"
EXT_3K = "results/external_validation_3k/external_validation_3k_summary.csv"
OUT_PATH = "results/figures/robustness_cross_donor.png"

os.makedirs("results/figures", exist_ok=True)


def normalise_model_name(m):
    u = str(m).strip().upper()
    if u in {"LOGISTIC REGRESSION", "LOGREG", "LR"}:
        return "LR"
    if u in {"XGBOOST", "XGB"}:
        return "XGB"
    if u in {"SANN"}:
        return "SANN"
    return u


# ── Load 68K splits ──
pca_splits = pd.read_csv(PCA_SPLITS)
hvg_splits = pd.read_csv(HVG_SPLITS)

pca_splits["model"] = pca_splits["model"].apply(normalise_model_name)
hvg_splits["model"] = hvg_splits["model"].apply(normalise_model_name)

# ── Load cross-donor ──
ext_8k = pd.read_csv(EXT_8K)
ext_3k = pd.read_csv(EXT_3K)

# ── Build stats for 68K splits ──
models = ["LR", "XGB", "SANN"]
reps = ["PCA", "HVG"]

split_data = {"PCA": pca_splits, "HVG": hvg_splits}

stats = {}
for rep in reps:
    for m in models:
        vals = split_data[rep][split_data[rep]["model"] == m]["macro_f1"].values
        stats[(m, rep)] = {"mean": vals.mean(), "min": vals.min(), "max": vals.max()}

# ── Get cross-donor F1 ──
def get_ext_f1(df, model, rep):
    name = f"{model}_{rep}"
    row = df[df["Model"] == name]
    if row.empty:
        return None
    if "Macro-F1 (known)" in row.columns:
        return float(row["Macro-F1 (known)"].values[0])
    elif "Macro-F1" in row.columns:
        return float(row["Macro-F1"].values[0])
    return None


# ── Colours ──
colors = {
    "PCA": "#1f77b4",
    "HVG": "#ff7f0e",
}
marker_8k = "s"   # square
marker_3k = "D"   # diamond
marker_68k = "o"  # circle

# ── Compute y limits ──
all_vals = []
for rep in reps:
    for m in models:
        s = stats[(m, rep)]
        all_vals.extend([s["min"], s["max"]])
        f1_8k = get_ext_f1(ext_8k, m, rep)
        f1_3k = get_ext_f1(ext_3k, m, rep)
        if f1_8k is not None:
            all_vals.append(f1_8k)
        if f1_3k is not None:
            all_vals.append(f1_3k)

padding = 0.015
lo = max(0.0, min(all_vals) - padding)
hi = min(1.0, max(all_vals) + padding)

# ── Plot ──
fig, ax = plt.subplots(figsize=(12, 6.5))

x_positions = []
labels = []
x = 1
label_dx = 0.12
offset_y = 0.004

for m in models:
    for rep in reps:
        xp = x if rep == "PCA" else x + 0.4
        x_positions.append(xp)
        labels.append(f"{m}\n{rep}")

        s = stats[(m, rep)]
        color = colors[rep]

        # Min-max bar (68K splits)
        ax.vlines(xp, s["min"], s["max"], color=color, linewidth=5, alpha=0.25)

        # 68K mean dot
        ax.scatter(xp, s["mean"], color=color, s=110, edgecolor="black",
                   marker=marker_68k, zorder=5)

        # Projection lines
        ax.vlines(xp, lo, s["mean"], color=color, linestyle="--",
                  alpha=0.4, linewidth=1.0, zorder=0)
        ax.hlines(s["mean"], 0.5, xp, color=color, linestyle="--",
                  alpha=0.4, linewidth=1.0, zorder=0)

        # 68K mean/min/max labels
        text_x = xp + label_dx
        ax.annotate(f"{s['max']:.3f}", xy=(xp, s["max"]),
                    xytext=(text_x, s["max"] + offset_y),
                    ha="left", fontsize=8, color="grey",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
        ax.annotate(f"{s['mean']:.3f}", xy=(xp, s["mean"]),
                    xytext=(text_x, s["mean"]),
                    ha="left", fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))
        ax.annotate(f"{s['min']:.3f}", xy=(xp, s["min"]),
                    xytext=(text_x, s["min"] - offset_y),
                    ha="left", fontsize=8, color="grey",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

        # 8K cross-donor dot
        f1_8k = get_ext_f1(ext_8k, m, rep)
        if f1_8k is not None:
            ax.scatter(xp - 0.08, f1_8k, color="tab:red", s=90,
                       edgecolor="black", linewidths=0.5,
                       marker=marker_8k, zorder=6)

        # 3K cross-donor dot
        f1_3k = get_ext_f1(ext_3k, m, rep)
        if f1_3k is not None:
            ax.scatter(xp + 0.08, f1_3k, color="tab:green", s=90,
                       edgecolor="black", linewidths=0.5,
                       marker=marker_3k, zorder=6)

    x += 1.4

# ── Axes ──
ax.set_xticks(x_positions)
ax.set_xticklabels(labels, fontsize=10)
ax.set_xlim(0.5, max(x_positions) + 0.7)
ax.set_ylim(lo, hi)

ax.set_title("Robustness Across Repeated Splits and Cross-Donor Generalisation", fontsize=13)
ax.set_xlabel("Model and Representation", fontsize=12)
ax.set_ylabel("Macro-F1", fontsize=12)
ax.grid(False)

# ── Legend ──
handles = [
    plt.Line2D([0], [0], color=colors["PCA"], lw=3, linestyle="--", alpha=0.5, label="PCA"),
    plt.Line2D([0], [0], color=colors["HVG"], lw=3, linestyle="--", alpha=0.5, label="HVG"),
    plt.scatter([], [], color=color, s=80, edgecolor="black", marker=marker_68k, label="68K Mean (5 splits)"),
    plt.scatter([], [], color="tab:red", s=80, edgecolor="black", marker=marker_8k, label="8K (Donor B)"),
    plt.scatter([], [], color="tab:green", s=80, edgecolor="black", marker=marker_3k, label="3K (Donor C)"),
]
ax.legend(handles=handles, fontsize=10, loc="lower left")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved to {os.path.abspath(OUT_PATH)}")
