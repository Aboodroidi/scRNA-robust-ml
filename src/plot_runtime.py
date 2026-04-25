import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================
PCA_PATH = "results/full_train_all_pca/baseline_metrics_full.csv"
HVG_PATH = "results/full_train_all_hvg/baseline_metrics_full.csv"
OUT_FIG = "figures/runtime_pca_vs_hvg_log.png"

os.makedirs("figures", exist_ok=True)


# =========================
# LOAD DATA
# =========================
pca_df = pd.read_csv(PCA_PATH)
hvg_df = pd.read_csv(HVG_PATH)

# standardise column names
pca_df = pca_df.rename(columns={
    "Model": "model",
    "TrainTimeSeconds": "time"
})

hvg_df = hvg_df.rename(columns={
    "Model": "model",
    "TrainTimeSeconds": "time"
})

# add representation label
pca_df["rep"] = "PCA"
hvg_df["rep"] = "HVG"

df = pd.concat([pca_df, hvg_df], ignore_index=True)

# clean
df["model"] = df["model"].astype(str).str.upper().str.strip()
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df = df.dropna(subset=["time"])
df = df[df["time"] > 0]


# =========================
# ORDER + COLORS
# =========================
models = ["LR", "XGB", "SANN"]
colors = {
    "PCA": "tab:blue",
    "HVG": "tab:orange"
}


# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(8.5, 5.5))

x = np.arange(len(models))
width = 0.35

all_vals = []

for i, rep in enumerate(["PCA", "HVG"]):
    vals = []
    for m in models:
        row = df[(df["model"] == m) & (df["rep"] == rep)]
        if row.empty:
            raise ValueError(f"Missing runtime for model={m}, rep={rep}")
        vals.append(float(row["time"].iloc[0]))

    all_vals.extend(vals)
    xpos = x + (i - 0.5) * width

    ax.bar(
        xpos,
        vals,
        width=width,
        label=rep,
        color=colors[rep],
        alpha=0.85
    )

    # value labels
    for xi, v in zip(xpos, vals):
        ax.text(
            xi,
            v * 1.15,
            f"{v:.1f}s",
            ha="center",
            va="bottom",
            fontsize=10
        )


# =========================
# AXES
# =========================
ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_title("Training Runtime Comparison (PCA vs HVG)")
ax.set_xlabel("Model")
ax.set_ylabel("Training Time (seconds)")

ax.grid(True, axis="y", alpha=0.3)
ax.legend()

# optional: make top a bit looser so labels fit
ax.set_ylim(0, max(all_vals) * 1.25)


# =========================
# SAVE
# =========================
out_path = os.path.abspath(OUT_FIG)

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)


# =========================
# PRINT OUTPUT
# =========================
print("\n✅ Runtime figure saved:\n")
print(out_path)