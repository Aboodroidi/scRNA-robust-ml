"""
PCA Elbow Plot — Variance explained per principal component (PBMC 68K).
Generates two subplots:
  1. Individual variance explained per component (elbow curve)
     with shaded candidate-flattening zone (PC 70–100)
  2. Cumulative variance explained
     with annotated values at PC 50, PC 70, and PC 100
Both mark n=50 (red) and n=100 (green) with vertical dashed lines.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import scanpy as sc

# ── Load data ──
print("Loading PBMC 68K dataset...")
adata = sc.read_h5ad("data/processed/pbmc68k_labeled.h5ad")
print(f"  Shape: {adata.shape}")

# ── Reverse z-scoring to recover log-normalised expression ──
mean = adata.var["mean"].values.astype(np.float64)
std = adata.var["std"].values.astype(np.float64)

X_scaled = np.array(adata.X, dtype=np.float64)
X_lognorm = X_scaled * std[np.newaxis, :] + mean[np.newaxis, :]
print(f"  Recovered log-norm X: min={X_lognorm.min():.4f}, max={X_lognorm.max():.4f}")

# ── Fit PCA on log-normalised HVG expression (200 components) ──
n_components = 200
print(f"Fitting PCA with {n_components} components on log-normalised HVG expression...")
pca = PCA(n_components=n_components, random_state=42)
pca.fit(X_lognorm)

var_ratio = pca.explained_variance_ratio_ * 100  # as percentage
cumulative = np.cumsum(var_ratio)

# Print key values for dissertation
print(f"\n  === Key cumulative variance values ===")
print(f"  Variance explained by PC1:  {var_ratio[0]:.2f}%")
print(f"  Cumulative at PC 50:  {cumulative[49]:.2f}%")
print(f"  Cumulative at PC 70:  {cumulative[69]:.2f}%")
print(f"  Cumulative at PC 80:  {cumulative[79]:.2f}%")
print(f"  Cumulative at PC 100: {cumulative[99]:.2f}%")
print(f"  Cumulative at PC 150: {cumulative[149]:.2f}%")
print(f"  Cumulative at PC 200: {cumulative[199]:.2f}%")
print(f"  Per-component at PC 50:  {var_ratio[49]:.4f}%")
print(f"  Per-component at PC 70:  {var_ratio[69]:.4f}%")
print(f"  Per-component at PC 100: {var_ratio[99]:.4f}%")

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

components = np.arange(1, n_components + 1)

# ── Plot 1: Individual variance per component ──
ax1.plot(components, var_ratio, "b-", linewidth=1.5, zorder=2)
ax1.fill_between(components, var_ratio, alpha=0.12, color="blue")

# Shaded candidate-flattening zone (PC 70–100)
ax1.axvspan(70, 100, alpha=0.15, color="orange", label="Flattening zone (PC 70–100)")

# Vertical lines
ax1.axvline(x=50, color="red", linestyle="--", linewidth=1.2, label="n = 50 (chosen)")
ax1.axvline(x=100, color="green", linestyle="--", linewidth=1.2, label="n = 100")

ax1.set_xlabel("Principal Component", fontsize=12)
ax1.set_ylabel("Variance Explained (%)", fontsize=12)
ax1.set_title("Variance Explained per Principal Component\nPBMC 68K Dataset (2,000 HVGs)", fontsize=13)
ax1.legend(fontsize=10, loc="upper right")
ax1.set_xlim(1, 200)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)

# ── Plot 2: Cumulative variance ──
ax2.plot(components, cumulative, "b-", linewidth=1.5, zorder=2)
ax2.fill_between(components, cumulative, alpha=0.12, color="blue")

# Vertical lines at PC 50 and PC 100
ax2.axvline(x=50, color="red", linestyle="--", linewidth=1.2, label="n = 50 (chosen)")
ax2.axvline(x=100, color="green", linestyle="--", linewidth=1.2, label="n = 100")

# Horizontal reference lines
ax2.axhline(y=cumulative[49], color="red", linestyle=":", linewidth=0.8, alpha=0.5)
ax2.axhline(y=cumulative[69], color="orange", linestyle=":", linewidth=0.8, alpha=0.5)
ax2.axhline(y=cumulative[99], color="green", linestyle=":", linewidth=0.8, alpha=0.5)

# Annotate PC 50
ax2.annotate(
    f"PC 50: {cumulative[49]:.1f}%",
    xy=(50, cumulative[49]),
    xytext=(18, cumulative[49] + 6),
    fontsize=11,
    fontweight="bold",
    color="red",
    arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
)

# Annotate PC 70
ax2.annotate(
    f"PC 70: {cumulative[69]:.1f}%",
    xy=(70, cumulative[69]),
    xytext=(85, cumulative[69] - 6),
    fontsize=11,
    fontweight="bold",
    color="darkorange",
    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2),
)

# Annotate PC 100
ax2.annotate(
    f"PC 100: {cumulative[99]:.1f}%",
    xy=(100, cumulative[99]),
    xytext=(120, cumulative[99] - 6),
    fontsize=11,
    fontweight="bold",
    color="green",
    arrowprops=dict(arrowstyle="->", color="green", lw=1.2),
)

ax2.set_xlabel("Principal Component", fontsize=12)
ax2.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
ax2.set_title("Cumulative Variance Explained\nPBMC 68K Dataset (2,000 HVGs)", fontsize=13)
ax2.legend(fontsize=10, loc="lower right")
ax2.set_xlim(1, 200)
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

outdir = "results/figures"
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "pca_elbow_plot.png")
plt.savefig(outpath, dpi=300, bbox_inches="tight")
print(f"\nSaved to {outpath}")
plt.close()
