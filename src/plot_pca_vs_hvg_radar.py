"""
Generate 2 radar diagrams for cross-donor Macro-F1 comparison:
  1. Donor B (8K) — PCA vs HVG
  2. Donor C (3K) — PCA vs HVG
Each has 3 model axes (LR, XGB, SANN) and 2 lines (PCA solid, HVG dashed).
For dissertation Section 4.1.2
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["LR", "XGB", "SANN"]

# Macro-F1 scores
f1_8k_pca = [0.954, 0.978, 0.985]
f1_8k_hvg = [0.903, 0.970, 0.842]
f1_3k_pca = [0.961, 0.954, 0.972]
f1_3k_hvg = [0.929, 0.953, 0.901]


def add_label(ax, theta, r, text, color, side="right"):
    PIXEL_LENGTH = 22
    if side == "right":
        dx = PIXEL_LENGTH
        ha = "left"
    else:
        dx = -PIXEL_LENGTH
        ha = "right"
    ax.annotate(
        text,
        xy=(theta, r),
        xytext=(dx, 0),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color=color, linewidth=1.4),
    )


def plot_radar(pca_vals, hvg_vals, title, filename):
    n = len(MODELS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    pca_closed = pca_vals + pca_vals[:1]
    hvg_closed = hvg_vals + hvg_vals[:1]

    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(MODELS, fontsize=12, fontweight="bold")

    ax.set_ylim(0.80, 1.0)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(["0.85", "0.90", "0.95", "1.00"], fontsize=9)
    ax.set_rlabel_position(180)

    pca_color = "#4C72B0"
    hvg_color = "#DD8452"

    # PCA
    ax.plot(angles_closed, pca_closed, color=pca_color, linewidth=2.2,
            marker="o", markersize=7, label="PCA")
    ax.fill(angles_closed, pca_closed, color=pca_color, alpha=0.10)

    # HVG
    ax.plot(angles_closed, hvg_closed, color=hvg_color, linewidth=2.2,
            marker="o", markersize=7, label="HVG")
    ax.fill(angles_closed, hvg_closed, color=hvg_color, alpha=0.06)

    # Labels
    for theta, pca_r, hvg_r in zip(angles, pca_vals, hvg_vals):
        add_label(ax, theta, pca_r, f"{pca_r:.3f}", pca_color, side="right")
        add_label(ax, theta, hvg_r, f"{hvg_r:.3f}", hvg_color, side="left")

    ax.set_title(title, size=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=11)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, filename)
    out_pdf = os.path.join(OUT_DIR, filename.replace(".png", ".pdf"))
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


# =========================
# GENERATE
# =========================
print()
plot_radar(f1_8k_pca, f1_8k_hvg,
           "Macro-F1: PCA vs HVG — 8K (Donor B)",
           "cross_donor_f1_8k_radar.png")

plot_radar(f1_3k_pca, f1_3k_hvg,
           "Macro-F1: PCA vs HVG — 3K (Donor C)",
           "cross_donor_f1_3k_radar.png")

print("\nDone.\n")
