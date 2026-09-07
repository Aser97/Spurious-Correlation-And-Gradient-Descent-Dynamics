"""
train_mlp_colored_mnist.py
--------------------------
Train a multi-layer MLP classifier on the colored-MNIST dataset and extract
penultimate-layer representations at multiple training checkpoints for
downstream SAE analysis.

Architecture:
    input (2352 = 28*28*3) -> 512 -> 256 -> 128 -> 2

The 128-dim penultimate layer is the representation phi(x) that will be
fed to the sparse autoencoder.

Outputs (in --out_dir):
    epoch_1/mlp_model.pt, epoch_1/representations.npz
    epoch_2/mlp_model.pt, epoch_2/representations.npz
    ...  (one subfolder per checkpoint epoch)

Usage (from Implicit-Bias-Spurious/):
    python train_mlp_colored_mnist.py --data_dir ./data_colored_mnist
    python train_mlp_colored_mnist.py --data_dir ./data_colored_mnist --checkpoint_epochs 1,2,3,4,5
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
#  MLP model
# ============================================================

class ColoredMNISTMLP(nn.Module):
    """
    4-layer MLP for binary classification on colored MNIST.

    input (2352) -> 512 -> 256 -> 128 -> 2

    The 128-dim output of the third hidden layer (after ReLU) is the
    representation phi(x) used for SAE analysis.
    """
    def __init__(self, d_in=2352, h1=512, h2=256, h3=128, n_classes=2):
        super().__init__()
        self.layer1 = nn.Linear(d_in, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, h3)
        self.head    = nn.Linear(h3, n_classes)
        self.relu    = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.head(x)

    def representation(self, x):
        """Return the 128-dim penultimate activations phi(x)."""
        with torch.no_grad():
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.relu(self.layer3(x))
        return x


# ============================================================
#  Training (one epoch at a time, to allow checkpointing)
# ============================================================

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch.  Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        total_correct += (logits.argmax(1) == y_batch).sum().item()
        total_n += X_batch.size(0)
    return total_loss / total_n, total_correct / total_n


def extract_and_save(model, X, metadata, save_dir, device):
    """Extract representations phi(x) for all samples and save to save_dir."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, "mlp_model.pt"))

    # Extract representations
    model.eval()
    N = X.shape[0]
    phi_list = []
    for i in range(0, N, 2048):
        batch = X[i:i+2048].to(device)
        phi_batch = model.representation(batch)
        phi_list.append(phi_batch.cpu().numpy())
    phi = np.concatenate(phi_list, axis=0)

    np.savez_compressed(
        os.path.join(save_dir, "representations.npz"),
        phi=phi,
        **metadata,
    )
    print(f"    phi shape: {phi.shape}, nonzero: {(phi > 0).mean():.3f}")
    return phi


def evaluate_per_group(model, X, y, groups, device):
    """Print accuracy per group and overall."""
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        preds = logits.argmax(1).cpu()

    for g in [0, 1]:
        mask = (groups == g)
        if mask.any():
            acc = (preds[mask] == y[mask]).float().mean().item()
            label = "majority (maroon)" if g == 0 else "minority (teal)"
            print(f"  {label}: acc={acc:.4f}  (n={mask.sum().item()})")

    acc_all = (preds == y).float().mean().item()
    print(f"  overall: acc={acc_all:.4f}")
    return acc_all


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP on colored MNIST, extract representations "
                    "at multiple checkpoints.")
    parser.add_argument("--data_dir", type=str, default="./data_colored_mnist",
                        help="Directory containing colored_mnist.npz")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: same as data_dir)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoint_epochs", type=str, default="1,2,3,4,5",
                        help="Comma-separated list of epochs at which to save "
                             "model + representations (default: 1,2,3,4,5)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.data_dir
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_epochs = set(int(e) for e in args.checkpoint_epochs.split(","))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    #  Load dataset
    # ------------------------------------------------------------------
    print("[1/3] Loading dataset...")
    data = np.load(os.path.join(args.data_dir, "colored_mnist.npz"))
    images = data["images"]          # (N, 28, 28, 3) uint8
    y      = data["y"]               # (N,) {0, 1}
    groups = data["groups"]          # (N,) {0, 1}
    digit_classes = data["digit_classes"]
    intensity_used = data["intensity_used"]

    N = len(images)
    print(f"  N={N}")

    # Flatten and normalize to [0, 1]
    X = images.reshape(N, -1).astype(np.float32) / 255.0   # (N, 2352)
    X = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()
    groups_t = torch.from_numpy(groups).long()

    # Metadata dict saved alongside each representation checkpoint
    metadata = dict(y=y, groups=groups, digit_classes=digit_classes,
                    intensity_used=intensity_used)

    # Train/test split (use first 60k for train, last 10k for test)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y_t[:60000], y_t[60000:]
    groups_train = groups_t[:60000]
    groups_test  = groups_t[60000:]

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # ------------------------------------------------------------------
    #  Train MLP with checkpointing
    # ------------------------------------------------------------------
    print(f"\n[2/3] Training MLP ({args.epochs} epochs, lr={args.lr})...")
    print(f"  Checkpoints at epochs: {sorted(ckpt_epochs)}")
    d_in = X.shape[1]
    model = ColoredMNISTMLP(d_in=d_in).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        avg_loss, acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)

        if epoch % 5 == 0 or epoch == 1 or epoch in ckpt_epochs:
            print(f"  Epoch {epoch:>3d}/{args.epochs}  "
                  f"loss={avg_loss:.4f}  acc={acc:.4f}")

        if epoch in ckpt_epochs:
            edir = os.path.join(args.out_dir, f"epoch_{epoch}")
            print(f"  --- Checkpoint epoch {epoch} ---")
            print("    Train set:")
            evaluate_per_group(model, X_train, y_train, groups_train, device)
            print("    Test set:")
            evaluate_per_group(model, X_test, y_test, groups_test, device)
            extract_and_save(model, X, metadata, edir, device)
            print(f"    Saved to {edir}")

    # ------------------------------------------------------------------
    #  Final evaluation
    # ------------------------------------------------------------------
    print(f"\n[3/3] Final evaluation (epoch {args.epochs})...")
    print("  Train set:")
    evaluate_per_group(model, X_train, y_train, groups_train, device)
    print("  Test set:")
    evaluate_per_group(model, X_test, y_test, groups_test, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
