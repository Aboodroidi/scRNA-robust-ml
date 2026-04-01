import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================
PCA_PATH = "results/full_train_all_pca/baseline_metrics_full.csv"
HVG_PATH = "results/full_train_all_hvg/baseline_metrics_full.csv"
OUT_DIR = "figures"

os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
pca_df = pd.read_csv(PCA_PATH)
hvg_df = pd.read_csv(HVG_PATH)

MODELS = ["LR", "XGB", "SANN"]


def extract_metric(df: pd.DataFrame, metric: str):
    vals = []
    for model in MODELS:
        row = df[df["Model"] == model]
        if row.empty:
            raise ValueError(f"Model '{model}' not found in CSV.")
        vals.append(float(row[metric].iloc[0]))
    return vals


pca_f1 = extract_metric(pca_df, "Macro-F1")
hvg_f1 = extract_metric(hvg_df, "Macro-F1")

pca_acc = extract_metric(pca_df, "Accuracy")
hvg_acc = extract_metric(hvg_df, "Accuracy")


# =========================
# LABEL FUNCTION
# =========================
def add_label(ax, theta, r, text, color, side="right"):
    """
    Fixed-length horizontal labels:
    - PCA (blue) goes to the right
    - HVG (orange) goes to the left
    """

    PIXEL_LENGTH = 22

    if side == "right":
        dx = PIXEL_LENGTH
        ha = "left"
    else:
        dx = -PIXEL_LENGTH
        ha = "right"

    dy = 0

    ax.annotate(
        text,
        xy=(theta, r),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=9,
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            linewidth=1.4,
        ),
    )


# =========================
# RADAR FUNCTION
# =========================
def plot_radar(pca_vals, hvg_vals, title, filename):
    n = len(MODELS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    pca_closed = pca_vals + pca_vals[:1]
    hvg_closed = hvg_vals + hvg_vals[:1]

    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw=dict(polar=True))

    # Start at 9 o'clock and go clockwise
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)

    # Axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels(MODELS, fontsize=11)

    # Radial scale
    ax.set_ylim(0.8, 1.0)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(["0.85", "0.90", "0.95", "1.00"], fontsize=9)
    ax.set_rlabel_position(180)

    pca_color = "tab:blue"
    hvg_color = "tab:orange"

    # PCA
    ax.plot(
        angles_closed,
        pca_closed,
        color=pca_color,
        linewidth=2.2,
        marker="o",
        markersize=7,
        label="PCA",
    )
    ax.fill(angles_closed, pca_closed, color=pca_color, alpha=0.10)

    # HVG
    ax.plot(
        angles_closed,
        hvg_closed,
        color=hvg_color,
        linewidth=2.2,
        marker="o",
        markersize=7,
        label="HVG",
    )
    ax.fill(angles_closed, hvg_closed, color=hvg_color, alpha=0.06)

    # Labels
    for theta, pca_r, hvg_r in zip(angles, pca_vals, hvg_vals):
        add_label(ax, theta, pca_r, f"{pca_r:.3f}", pca_color, side="right")
        add_label(ax, theta, hvg_r, f"{hvg_r:.3f}", hvg_color, side="left")

    plt.title(title, size=13, pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10))

    out_path = os.path.abspath(os.path.join(OUT_DIR, filename))
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return out_path


# =========================
# GENERATE FIGURES
# =========================
macro_path = plot_radar(
    pca_f1,
    hvg_f1,
    "Model Comparison (Macro-F1): PCA vs HVG",
    "pca_vs_hvg_macro_f1_radar.png",
)

acc_path = plot_radar(
    pca_acc,
    hvg_acc,
    "Model Comparison (Accuracy): PCA vs HVG",
    "pca_vs_hvg_accuracy_radar.png",
)


# =========================
# PRINT OUTPUT
# =========================
print("\n✅ Figures saved successfully:\n")
print(f"Macro-F1 Radar: {macro_path}")
print(f"Accuracy Radar: {acc_path}\n")