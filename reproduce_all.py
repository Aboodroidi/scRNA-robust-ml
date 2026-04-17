#!/usr/bin/env python
"""
reproduce_all.py — End-to-end orchestration of the scRNA-robust-ml dissertation pipeline.

Usage:
    python reproduce_all.py [flags]

Stages (in order):
    1. Preprocessing  — 68K, 8K, 3K PBMC datasets → annotated .h5ad files
    2. Baseline + SANN training (HVG)
    3. Baseline + SANN training (PCA)
    4. Cross-donor evaluation (8K, 3K)
    5. Calibration (temperature scaling, reliability diagrams)
    6. Ablation (mask on/off, PCA and HVG)
    7. Robustness (5-seed cross-donor splits)
    8. Figure generation

Flags:
    --skip-preprocess    Skip stage 1 (use existing .h5ad files in data/processed/)
    --skip-training      Skip stages 2–3 (use existing trained models)
    --skip-evaluation    Skip stage 4
    --skip-calibration   Skip stage 5
    --skip-ablation      Skip stage 6
    --skip-robustness    Skip stage 7 (~20 min)
    --skip-figures       Skip stage 8
    --figures-only       Run only stage 8
    --full-retrain       Force retrain even if models exist (~4 hours)
    --dry-run            Print stages without executing

By default, skips training if trained models already exist on disk.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ══════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════
REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

HVG_DIR = REPO_ROOT / "results" / "full_train_all_hvg_coarse"
PCA_DIR = REPO_ROOT / "results" / "full_train_all_pca_coarse"

ENV = os.environ.copy()
ENV["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════
def log(msg, stage=None, stages_total=8):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}]"
    if stage is not None:
        prefix += f" Stage {stage}/{stages_total}:"
    print(f"{prefix} {msg}", flush=True)


def run(cmd, dry_run=False, stage_label=None):
    """Run a shell command with logging."""
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    log(f"Running: {cmd_str}")
    if dry_run:
        return 0
    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=ENV)
    elapsed = time.time() - t0
    log(f"  done in {elapsed:.1f}s")
    if result.returncode != 0:
        log(f"  ⚠️  command exited with code {result.returncode}")
    return result.returncode


def exists(path):
    return (REPO_ROOT / path).exists()


def trained_models_exist():
    return (HVG_DIR / "lr_model.pkl").exists() and (PCA_DIR / "lr_model.pkl").exists()


def ensure_dirs():
    for d in ["results", "results/figures", "data/processed", "models"]:
        (REPO_ROOT / d).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════
# STAGES
# ══════════════════════════════════════════════════
def stage_preprocess(args):
    log("Preprocessing 68K, 8K, 3K PBMC datasets", stage=1)
    if args.skip_preprocess:
        log("  skipped (--skip-preprocess)")
        return
    if all(exists(f"data/processed/{f}") for f in
           ["pbmc68k_labeled.h5ad", "pbmc8k_labeled.h5ad", "pbmc3k_labeled.h5ad"]):
        log("  all .h5ad files already exist — skipping (use --full-retrain to force)")
        if not args.full_retrain:
            return

    if not exists("data/processed/pbmc68k_labeled.h5ad") or args.full_retrain:
        run([PYTHON, "src/preprocess.py"], args.dry_run)
        run([PYTHON, "src/annotate.py"], args.dry_run)
    if not exists("data/processed/pbmc8k_labeled.h5ad") or args.full_retrain:
        run([PYTHON, "src/preprocess_pbmc8k.py"], args.dry_run)
    if not exists("data/processed/pbmc3k_labeled.h5ad") or args.full_retrain:
        run([PYTHON, "src/preprocess_pbmc3k.py"], args.dry_run)


def stage_train_hvg(args):
    log("Training baselines + SANN on HVG features (~3h for XGB_HVG)", stage=2)
    if args.skip_training:
        log("  skipped (--skip-training)")
        return
    if (HVG_DIR / "xgb_model.json").exists() and not args.full_retrain:
        log("  HVG models exist — skipping (use --full-retrain to force)")
        return
    run([
        PYTHON, "src/train_all_full_models.py",
        "--label_key", "cell_type_coarse",
        "--outdir", str(HVG_DIR),
        "--n_seeds", "3",
    ], args.dry_run)


def stage_train_pca(args):
    log("Training baselines + SANN on PCA features", stage=3)
    if args.skip_training:
        log("  skipped (--skip-training)")
        return
    if (PCA_DIR / "xgb_model.json").exists() and not args.full_retrain:
        log("  PCA models exist — skipping (use --full-retrain to force)")
        return
    run([
        PYTHON, "src/train_all_pca_models.py",
        "--label_key", "cell_type_coarse",
        "--outdir", str(PCA_DIR),
        "--n_seeds", "3",
    ], args.dry_run)


def stage_evaluate(args):
    log("Cross-donor evaluation on 8K and 3K datasets", stage=4)
    if args.skip_evaluation:
        log("  skipped (--skip-evaluation)")
        return
    run([
        PYTHON, "src/evaluate_external_pbmc8k.py",
        "--label_key", "cell_type_coarse",
        "--hvg_dir", str(HVG_DIR),
        "--pca_dir", str(PCA_DIR),
    ], args.dry_run)
    run([
        PYTHON, "src/evaluate_external_pbmc3k.py",
        "--label_key", "cell_type_coarse",
        "--hvg_dir", str(HVG_DIR),
        "--pca_dir", str(PCA_DIR),
    ], args.dry_run)


def stage_calibration(args):
    log("Calibration: temperature scaling + reliability diagrams", stage=5)
    if args.skip_calibration:
        log("  skipped (--skip-calibration)")
        return
    run([
        PYTHON, "src/plot_temp_scaling_all.py",
        "--model_dir", "results/full_train_all_pca",
        "--out", "results/figures/temp_scaling_all_pca.png",
    ], args.dry_run)
    run([
        PYTHON, "src/plot_reliability_all_models.py",
        "--model_dir", "results/full_train_all_pca",
        "--out", "results/figures/reliability_all_models_pca.png",
    ], args.dry_run)


def stage_ablation(args):
    log("Ablation: mask on/off (PCA and HVG)", stage=6)
    if args.skip_ablation:
        log("  skipped (--skip-ablation)")
        return
    run([
        PYTHON, "src/ablation_mask_pca.py",
        "--label_key", "cell_type_coarse",
        "--outdir", "results/ablation_mask",
        "--mode", "both",
    ], args.dry_run)


def stage_robustness(args):
    log("Robustness: 5-seed cross-donor splits (~20 min)", stage=7)
    if args.skip_robustness:
        log("  skipped (--skip-robustness)")
        return
    run([PYTHON, "src/robustness_cross_donor.py"], args.dry_run)


def stage_figures(args):
    log("Generating figures", stage=8)
    if args.skip_figures:
        log("  skipped (--skip-figures)")
        return

    figures = [
        (["src/plot_cell_type_distribution.py"], "Cell-type composition"),
        (["src/plot_marker_dotplot.py"], "Marker gene dotplot"),
        (["src/plot_pca_elbow.py"], "PCA variance elbow"),
        (["src/plot_umap.py"], "UMAP 2D"),
        (["src/plot_umap_3d.py"], "UMAP 3D"),
        (["src/plot_umap_true_labels.py"], "UMAP true labels"),
        (["src/plot_umap_pred_sann.py"], "UMAP SANN predictions"),
        (["src/plot_umap_misclassifications.py"], "UMAP misclassifications"),
        (["src/plot_umap_misclassified_pairs_sann.py"], "UMAP misclassified pairs"),
        (["src/plot_misclassified_2d.py"], "Misclassified 2D"),
        (["src/plot_misclassified_3d.py"], "Misclassified 3D"),
        (["src/plot_confusion_triptych.py"], "Confusion matrix triptych"),
        (["src/plot_confidence_histograms.py"], "Confidence histograms"),
        (["src/plot_per_class_heatmap.py"], "Per-class F1 heatmap"),
        (["src/plot_convergence_curves_option2.py"], "Convergence curves"),
        (["src/plot_library_size_boxplot.py"], "Library size distribution"),
        (["src/plot_pca_vs_hvg_radar.py"], "PCA vs HVG radar"),
        (["src/plot_pca_cross_donor_bar.py"], "PCA cross-donor bar chart"),
        (["src/plot_robustness.py"], "Robustness 68K"),
        (["src/plot_robustness_cross_donor.py"], "Robustness cross-donor"),
        (["src/plot_runtime.py"], "Runtime comparison"),
    ]
    for cmd, label in figures:
        if not (REPO_ROOT / cmd[0]).exists():
            log(f"  skipping (missing): {label}")
            continue
        log(f"  → {label}")
        run([PYTHON] + cmd, args.dry_run)


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="End-to-end orchestration for scRNA-robust-ml pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-evaluation", action="store_true")
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--full-retrain", action="store_true",
                    help="Force retrain even if models exist (~4 hours)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stages without executing")
    args = ap.parse_args()

    if args.figures_only:
        args.skip_preprocess = True
        args.skip_training = True
        args.skip_evaluation = True
        args.skip_calibration = True
        args.skip_ablation = True
        args.skip_robustness = True

    ensure_dirs()

    log("════════════════════════════════════════════════════════════")
    log("  scRNA-robust-ml: reproduce_all.py")
    log("════════════════════════════════════════════════════════════")
    if args.dry_run:
        log("  DRY RUN — no commands will execute")

    t_start = time.time()

    stage_preprocess(args)
    stage_train_hvg(args)
    stage_train_pca(args)
    stage_evaluate(args)
    stage_calibration(args)
    stage_ablation(args)
    stage_robustness(args)
    stage_figures(args)

    total = time.time() - t_start
    log("════════════════════════════════════════════════════════════")
    log(f"  Pipeline complete — total wall-clock: {total/60:.1f} min")
    log(f"  Main results:   results/external_validation/external_validation_summary.csv")
    log(f"                  results/external_validation_3k/external_validation_3k_summary.csv")
    log(f"  Figures:        results/figures/")
    log("════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
