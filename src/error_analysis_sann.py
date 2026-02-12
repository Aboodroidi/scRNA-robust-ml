# src/error_analysis_sann.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="SANN error analysis: top confusions + per-class F1 bar plot + submatrix.")
    p.add_argument("--report_csv", type=str, default="results/reports/sann_classification_report.csv",
                   help="Per-class report produced by eval_sann.py (classification_report output_dict).")
    p.add_argument("--y_true", type=str, default="results/sann_test_true.npy")
    p.add_argument("--y_pred", type=str, default="results/sann_test_pred.npy")

    p.add_argument("--out_reports", type=str, default="results/reports")
    p.add_argument("--out_figures", type=str, default="results/figures")

    p.add_argument("--top_k", type=int, default=5, help="Top-K off-diagonal confusion pairs to extract.")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def load_class_names_from_report(report_csv: str):
    """
    Assumes sann_classification_report.csv is a DataFrame where index contains class names
    plus rows like 'accuracy', 'macro avg', 'weighted avg'.
    """
    df = pd.read_csv(report_csv, index_col=0)
    # keep only rows that look like classes (have numeric precision/recall/f1/support)
    # In sklearn output_dict -> DataFrame, class rows have support present.
    class_rows = df[~df.index.isin(["accuracy", "macro avg", "weighted avg"])]
    class_names = list(class_rows.index)
    return df, class_rows, class_names


def extract_top_confusions(y_true, y_pred, class_names, top_k=5):
    """
    Returns a DataFrame of top off-diagonal confusions by count.
    Includes:
      true_label, pred_label, count, pct_of_true_class
    """
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

    # zero diagonal
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

    # get top entries
    flat_idx = np.argsort(cm_off.ravel())[::-1]  # descending
    results = []

    true_counts = cm.sum(axis=1)  # row totals = true class size

    taken = 0
    for idx in flat_idx:
        if taken >= top_k:
            break
        i = idx // n_classes
        j = idx % n_classes
        count = cm_off[i, j]
        if count <= 0:
            break
        pct = (count / true_counts[i]) * 100 if true_counts[i] > 0 else 0.0
        results.append({
            "true_label": class_names[i],
            "pred_label": class_names[j],
            "count": int(count),
            "pct_of_true_class": round(float(pct), 3),
            "true_class_size": int(true_counts[i]),
        })
        taken += 1

    return pd.DataFrame(results), cm


def plot_per_class_f1(class_rows: pd.DataFrame, out_path: str, dpi: int = 300):
    """
    Bar plot of per-class F1, sorted ascending.
    Highlight bottom 3 in a different colour.
    """
    # Expect columns: precision, recall, f1-score, support
    if "f1-score" not in class_rows.columns:
        raise ValueError("Expected 'f1-score' column in classification report CSV.")

    df = class_rows.copy()
    df["f1-score"] = pd.to_numeric(df["f1-score"], errors="coerce")
    df = df.dropna(subset=["f1-score"])
    df = df.sort_values("f1-score", ascending=True)

    labels = df.index.tolist()
    f1_vals = df["f1-score"].values

    # highlight bottom 3
    n = len(df)
    colors = ["#C0C0C0"] * n  # light grey for most
    for k in range(min(3, n)):
        colors[k] = "#404040"  # darker grey for bottom classes (no “rainbow”, still muted)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(range(n), f1_vals, color=colors)

    ax.set_ylabel("F1 score")
    ax.set_xlabel("Cell type")
    ax.set_title("SANN Per-Class F1 Scores (Test Set)")
    ax.set_ylim(0, 1.0)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_submatrix(cm: np.ndarray, class_names, true_label: str, pred_label: str, out_path: str, dpi: int = 300):
    """
    Plot a 2x2 submatrix for a pair (A,B): true A/B vs pred A/B (raw counts).
    """
    idx = {name: i for i, name in enumerate(class_names)}
    if true_label not in idx or pred_label not in idx:
        raise ValueError("Labels not found in class_names for submatrix plot.")

    a = idx[true_label]
    b = idx[pred_label]

    sub = cm[np.ix_([a, b], [a, b])]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(sub, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([true_label, pred_label], rotation=45, ha="right")
    ax.set_yticklabels([true_label, pred_label])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Submatrix: {true_label} vs {pred_label}")

    # annotate counts
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(sub[i, j])), ha="center", va="center", fontsize=11)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    safe_makedirs(args.out_reports)
    safe_makedirs(args.out_figures)

    # Load report + class names
    report_df, class_rows, class_names = load_class_names_from_report(args.report_csv)

    # Load y arrays
    y_true = np.load(args.y_true).astype(int)
    y_pred = np.load(args.y_pred).astype(int)

    print(f"[Sanity] y_true length: {len(y_true)} | y_pred length: {len(y_pred)}")
    print(f"[Sanity] #classes from report: {len(class_names)}")
    print(f"[Sanity] class names: {class_names}")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")

    if y_true.min() < 0 or y_true.max() >= len(class_names):
        raise ValueError("y_true values out of range for class_names.")
    if y_pred.min() < 0 or y_pred.max() >= len(class_names):
        raise ValueError("y_pred values out of range for class_names.")

    # Step 1: Top confused pairs
    top_df, cm = extract_top_confusions(y_true, y_pred, class_names, top_k=args.top_k)
    top_path = os.path.join(args.out_reports, "sann_top_confusions.csv")
    top_df.to_csv(top_path, index=False)
    print(f"[Saved] {top_path}")
    print("[Top confusions]")
    print(top_df)

    # Step 2: Per-class F1 bar plot (main fpython src/error_analysis_sann.pyigure)
    f1_fig_path = os.path.join(args.out_figures, "sann_per_class_f1.png")
    plot_per_class_f1(class_rows, f1_fig_path, dpi=args.dpi)
    print(f"[Saved] {f1_fig_path}")

    # Step 3: Confusion submatrix for top pair (optional strong)
    if len(top_df) > 0:
        a = top_df.iloc[0]["true_label"]
        b = top_df.iloc[0]["pred_label"]
        sub_fig_path = os.path.join(args.out_figures, "sann_top_pair_submatrix.png")
        plot_submatrix(cm, class_names, a, b, sub_fig_path, dpi=args.dpi)
        print(f"[Saved] {sub_fig_path}")
    else:
        print("[Info] No off-diagonal confusions found (unexpected). Skipping submatrix plot.")


if __name__ == "__main__":
    main()