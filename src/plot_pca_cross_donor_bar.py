"""
Generate grouped bar chart: PCA-based model performance across cross-donor datasets.
For dissertation Figure (Table 1 replacement) — Section 4.1.1
"""
import matplotlib.pyplot as plt
import numpy as np

# --- Data: PCA models only ---
models = ["LR_PCA", "XGB_PCA", "SANN_PCA\n(ensemble)"]

f1_8k = [0.954, 0.978, 0.985]
f1_3k = [0.961, 0.954, 0.972]

x = np.arange(len(models))
width = 0.30

fig, ax = plt.subplots(figsize=(8, 5))

bars_8k = ax.bar(x - width/2, f1_8k, width, label="8K (Donor B)", color="#4C72B0", edgecolor="black", linewidth=0.6)
bars_3k = ax.bar(x + width/2, f1_3k, width, label="3K (Donor C)", color="#DD8452", edgecolor="black", linewidth=0.6)

# Add value labels on top of each bar
for bar in bars_8k:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.002, f"{h:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

for bar in bars_3k:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.002, f"{h:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("Macro-F1 Score", fontsize=12)
ax.set_title("PCA-Based Classification: Cross-Donor Macro-F1", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0.92, 1.005)
ax.legend(fontsize=11, loc="lower left")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/figures/pca_cross_donor_f1.png", dpi=300, bbox_inches="tight")
plt.savefig("results/figures/pca_cross_donor_f1.pdf", dpi=300, bbox_inches="tight")
print("Saved to results/figures/pca_cross_donor_f1.png and .pdf")
plt.show()
