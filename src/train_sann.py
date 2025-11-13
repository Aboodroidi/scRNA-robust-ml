# src/train_sann.py
import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_numpy(arr):
    return arr.toarray() if issparse(arr) else np.asarray(arr)

# ---------------------------
# Model
# ---------------------------
class SANN(nn.Module):
    """
    Sparse-Aware Neural Network (SANN)
    Linear -> BN -> GELU -> Dropout -> Linear -> BN -> GELU -> Dropout -> Linear
    L1 penalty on the first Linear layer to encourage feature sparsity
    """
    def __init__(self, input_dim, num_classes, hidden=256, dropout=0.3, l1_lambda=1e-5):
        super().__init__()
        self.l1_lambda = l1_lambda

        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.act2 = nn.GELU()
        self.do2 = nn.Dropout(dropout)

        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.do1(self.act1(self.bn1(self.fc1(x))))
        x = self.do2(self.act2(self.bn2(self.fc2(x))))
        return self.head(x)

    def l1_penalty(self):
        return self.l1_lambda * torch.abs(self.fc1.weight).sum()

# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb) + model.l1_penalty()
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred)
        labels.append(yb.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1

# ---------------------------
# Main
# ---------------------------
def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    # Load data
    adata = sc.read_h5ad(args.input)
    X = adata.X
    y_cat = adata.obs[args.label_col].astype("category")
    y = y_cat.cat.codes
    label_names = list(y_cat.cat.categories)

    # Optional downsample for quick debugging
    if args.max_cells is not None and args.max_cells > 0 and X.shape[0] > args.max_cells:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(X.shape[0], size=args.max_cells, replace=False)
        if issparse(X):
            X = X[idx]
        else:
            X = np.asarray(X)[idx]
        y = np.asarray(y)[idx]
        print(f"Subsampled to {X.shape[0]} cells for debugging.")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # Scale features (handle sparse vs dense properly)
    scaler = StandardScaler(with_mean=not issparse(X_train))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Convert to tensors (ensure dense arrays)
    X_train_t = torch.tensor(to_numpy(X_train), dtype=torch.float32)
    X_test_t  = torch.tensor(to_numpy(X_test),  dtype=torch.float32)
    y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.long)
    y_test_t  = torch.tensor(np.asarray(y_test),  dtype=torch.long)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=args.batch_size)

    # Model, loss, optimiser
    input_dim = X_train_t.shape[1]
    model = SANN(
        input_dim=input_dim,
        num_classes=len(label_names),
        hidden=args.hidden,
        dropout=args.dropout,
        l1_lambda=args.l1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train
    losses = []
    best_f1 = -1.0
    patience_count = 0
    os.makedirs(args.outdir, exist_ok=True)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        losses.append((epoch, loss))
        acc, f1 = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d} | loss {loss:.4f} | test_acc {acc:.4f} | test_f1 {f1:.4f}")

        # simple early stopping on macro-F1
        if args.early_stop_patience > 0:
            if f1 > best_f1 + 1e-4:
                best_f1 = f1
                patience_count = 0
                torch.save(model.state_dict(), os.path.join(args.outdir, "sann_best.pt"))
            else:
                patience_count += 1
                if patience_count >= args.early_stop_patience:
                    print("Early stopping triggered.")
                    break

    # Final eval and save artefacts
    if args.early_stop_patience > 0 and os.path.exists(os.path.join(args.outdir, "sann_best.pt")):
        model.load_state_dict(torch.load(os.path.join(args.outdir, "sann_best.pt"), map_location=device))

    final_acc, final_f1 = evaluate(model, test_loader, device)
    print(f"SANN Final | test_acc {final_acc:.4f} | test_f1 {final_f1:.4f}")

    torch.save(model.state_dict(), os.path.join(args.outdir, "sann_final.pt"))
    pd.DataFrame(losses, columns=["epoch", "train_loss"]).to_csv(
        os.path.join(args.outdir, "sann_train_loss.csv"), index=False
    )
    pd.DataFrame(
        {"metric": ["accuracy", "macro_f1"], "value": [final_acc, final_f1]}
    ).to_csv(os.path.join(args.outdir, "sann_metrics.csv"), index=False)

    # Save labels and config
    with open(os.path.join(args.outdir, "labels.txt"), "w") as f:
        for name in label_names:
            f.write(f"{name}\n")
    with open(os.path.join(args.outdir, "sann_config.txt"), "w") as f:
        f.write(
            f"input={args.input}\nlabel_col={args.label_col}\n"
            f"hidden={args.hidden}\ndropout={args.dropout}\nl1={args.l1}\n"
            f"lr={args.lr}\nweight_decay={args.weight_decay}\n"
            f"batch_size={args.batch_size}\nepochs={args.epochs}\n"
            f"test_size={args.test_size}\nseed={args.seed}\n"
            f"max_cells={args.max_cells}\n"
            f"early_stop_patience={args.early_stop_patience}\n"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/pbmc68k_labeled.h5ad", help="Path to labeled .h5ad")
    parser.add_argument("--label_col", default="cell_type", help="Obs column with labels")
    parser.add_argument("--outdir", default="results/sann", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--l1", type=float, default=1e-5, help="L1 penalty on first layer")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_cells", type=int, default=0, help="Optional subsample size for quick runs")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="0 disables early stopping")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = parser.parse_args()
    main(args)