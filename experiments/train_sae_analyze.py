"""
train_sae_analyze.py
--------------------
Train a Sparse Autoencoder (SAE) on the MLP representations phi(x) from the
colored-MNIST experiment, then analyze whether spurious correlation structure
is preserved in the learned representation.

Pipeline:
  1. Load representations phi(x) and metadata.
  2. Train a tied-weight SAE:  c = ReLU(M phi + b),  phi_hat = M^T c
     with loss = ||phi - phi_hat||^2 + alpha * ||c||_1
  3. Classify each SAE feature as r-related or s-related using
     group-conditional correlations with digit class:
       rho_0(k) = corr(c_k, d | g=0)   (within majority)
       rho_1(k) = corr(c_k, d | g=1)   (within minority)
     Same sign   -> r-feature (tracks digit, invariant to color)
     Opposite sign -> s-feature (tracks color, flips with group)
  4. Decompose phi into phi_r and phi_s components.
  5. Fit a linear map phi_r -> phi_s per group and measure R^2.
  6. (Optional) Permutation test: shuffle phi_s rows within each group
     to obtain a null distribution of R^2, yielding a p-value.
  7. Produce figures.

Can process a single directory or loop over all epoch_* subdirs:
    python train_sae_analyze.py --data_dir ./data_colored_mnist/epoch_1
    python train_sae_analyze.py --data_dir ./data_colored_mnist --run_all_epochs

Reference: Cunningham et al. (2023) "Sparse Autoencoders Find Highly
Interpretable Features in Language Models" (arXiv:2309.08600).
"""

import os
import glob
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
#  Sparse Autoencoder (tied weights, following the SAE paper)
# ============================================================

class SparseAutoencoder(nn.Module):
    """
    Tied-weight sparse autoencoder.

        c     = ReLU(M x + b)          (encoding)
        x_hat = M^T c                  (decoding with tied weights)

    The dictionary consists of the rows of M (each row f_k is a feature
    direction in the activation space).  The hidden dimension d_hid is
    the dictionary size and should be > d_in for an overcomplete basis.

    Loss = ||x - x_hat||^2  +  alpha * ||c||_1
    """

    def __init__(self, d_in, d_hid):
        super().__init__()
        self.d_in = d_in
        self.d_hid = d_hid

        # M: (d_hid, d_in) — rows are dictionary features
        self.M = nn.Parameter(torch.randn(d_hid, d_in) * 0.01)
        self.b = nn.Parameter(torch.zeros(d_hid))
        self.relu = nn.ReLU()

        # Normalize rows at init
        with torch.no_grad():
            self.M.div_(self.M.norm(dim=1, keepdim=True) + 1e-12)

    def encode(self, x):
        """Compute sparse coefficients c = ReLU(M x + b)."""
        return self.relu(x @ self.M.T + self.b)   # (batch, d_hid)

    def decode(self, c):
        """Reconstruct x_hat = M^T c."""
        return c @ self.M                          # (batch, d_in)

    def forward(self, x):
        c = self.encode(x)
        x_hat = self.decode(c)
        return x_hat, c

    def normalize_features(self):
        """Normalize dictionary rows to unit norm (prevents trivial sparsity
        reduction by scaling up M)."""
        with torch.no_grad():
            norms = self.M.norm(dim=1, keepdim=True)
            self.M.div_(norms + 1e-12)


# ============================================================
#  SAE training
# ============================================================

def train_sae(model, data_loader, epochs, lr, alpha, device, verbose=True):
    """
    Train SAE with reconstruction + L1 sparsity loss.

    Parameters
    ----------
    alpha : float
        Weight on the L1 sparsity penalty.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        total_recon, total_l1, total_n = 0.0, 0.0, 0
        for (X_batch,) in data_loader:
            X_batch = X_batch.to(device)
            x_hat, c = model(X_batch)

            recon_loss = ((X_batch - x_hat) ** 2).sum(dim=1).mean()
            l1_loss = c.abs().sum(dim=1).mean()
            loss = recon_loss + alpha * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Re-normalize dictionary rows after each step
            model.normalize_features()

            total_recon += recon_loss.item() * X_batch.size(0)
            total_l1 += l1_loss.item() * X_batch.size(0)
            total_n += X_batch.size(0)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            avg_r = total_recon / total_n
            avg_l1 = total_l1 / total_n
            # Fraction of variance unexplained
            print(f"  Epoch {epoch:>3d}/{epochs}  "
                  f"recon={avg_r:.4f}  L1={avg_l1:.4f}")

    return model


# ============================================================
#  Feature classification: r-related vs s-related
#  (group-conditional correlations — avoids the confound
#   where corr(c_k, intensity) ≈ (1-2ε)*corr(c_k, digit)
#   because intensity ≈ digit/9 for the majority group)
# ============================================================

def classify_features(c_all, digit_classes, groups, threshold=0.2):
    """
    Classify each SAE feature using group-conditional correlations
    with digit class.

    For each feature k compute:
      rho_maj(k) = corr(c_k, d | g=0)   — within majority
      rho_min(k) = corr(c_k, d | g=1)   — within minority

    Within majority: background intensity = d/9 (same direction as d).
    Within minority: background intensity = 1-d/9 (opposite direction).

    Therefore:
      same sign   rho_maj, rho_min  ->  r-feature (tracks digit shape)
      opposite sign                  ->  s-feature (tracks background color)

    Both group-conditional correlations must exceed `threshold` in
    absolute value for a feature to be classified as r or s.

    Parameters
    ----------
    c_all : ndarray (N, d_hid), SAE activations for all samples
    digit_classes : ndarray (N,), 0-9
    groups : ndarray (N,), 0=majority, 1=minority
    threshold : float, minimum |correlation| for significance

    Returns
    -------
    feature_type : list of str ('r', 's', 'weak', 'dead')
    rho_maj : ndarray (d_hid,), within-majority corr with digit class
    rho_min : ndarray (d_hid,), within-minority corr with digit class
    """
    d_hid = c_all.shape[1]
    rho_maj = np.zeros(d_hid)
    rho_min = np.zeros(d_hid)
    feature_type = []

    maj_mask = (groups == 0)
    min_mask = (groups == 1)
    d_maj = digit_classes[maj_mask].astype(np.float64)
    d_min = digit_classes[min_mask].astype(np.float64)

    for k in range(d_hid):
        ck = c_all[:, k]

        # Dead feature: never activates
        if ck.std() < 1e-8:
            feature_type.append("dead")
            continue

        ck_maj = ck[maj_mask]
        ck_min = ck[min_mask]

        # If the feature has near-zero variance in either group,
        # we cannot compute a meaningful correlation there
        if ck_maj.std() < 1e-8 or ck_min.std() < 1e-8:
            feature_type.append("weak")
            continue

        rho_maj[k] = np.corrcoef(ck_maj, d_maj)[0, 1]
        rho_min[k] = np.corrcoef(ck_min, d_min)[0, 1]

        abs_0 = abs(rho_maj[k])
        abs_1 = abs(rho_min[k])

        if min(abs_0, abs_1) >= threshold:
            if rho_maj[k] * rho_min[k] > 0:       # same sign
                feature_type.append("r")
            else:                                   # opposite sign
                feature_type.append("s")
        else:
            feature_type.append("weak")

    return feature_type, rho_maj, rho_min


# ============================================================
#  Decompose into phi_r and phi_s, measure linearity
# ============================================================

def decompose_and_measure(c_all, feature_type, groups):
    """
    Decompose phi(x) into r-activations and s-activations using the SAE
    feature classification, then fit a linear map phi_r -> phi_s per group.

    phi_r = [c_k for k where feature_type[k] == 'r']
    phi_s = [c_k for k where feature_type[k] == 's']

    Returns
    -------
    results : dict with R^2 scores, number of features, etc.
    """
    r_idx = [k for k, t in enumerate(feature_type) if t == "r"]
    s_idx = [k for k, t in enumerate(feature_type) if t == "s"]
    mixed_idx = [k for k, t in enumerate(feature_type) if t == "mixed"]
    dead_idx = [k for k, t in enumerate(feature_type) if t == "dead"]
    weak_idx = [k for k, t in enumerate(feature_type) if t == "weak"]

    print(f"\n  Feature classification:")
    print(f"    r-features:     {len(r_idx)}")
    print(f"    s-features:     {len(s_idx)}")
    print(f"    mixed features: {len(mixed_idx)}")
    print(f"    weak features:  {len(weak_idx)}")
    print(f"    dead features:  {len(dead_idx)}")

    if len(r_idx) == 0 or len(s_idx) == 0:
        print("  WARNING: not enough r or s features to measure linearity.")
        return {"r2_maj": None, "r2_min": None,
                "n_r": len(r_idx), "n_s": len(s_idx)}

    phi_r = c_all[:, r_idx]   # (N, n_r)
    phi_s = c_all[:, s_idx]   # (N, n_s)

    results = {"n_r": len(r_idx), "n_s": len(s_idx),
               "n_mixed": len(mixed_idx), "n_dead": len(dead_idx),
               "n_weak": len(weak_idx),
               "r_idx": r_idx, "s_idx": s_idx}

    # Fit linear map phi_r -> phi_s per group
    for g, name in [(0, "majority"), (1, "minority")]:
        mask = (groups == g)
        if mask.sum() < 10:
            results[f"r2_{name}"] = None
            continue

        probe = Ridge(alpha=1.0)
        probe.fit(phi_r[mask], phi_s[mask])
        phi_s_hat = probe.predict(phi_r[mask])
        r2 = r2_score(phi_s[mask], phi_s_hat,
                       multioutput="variance_weighted")
        results[f"r2_{name}"] = float(r2)
        print(f"    Linear map phi_r -> phi_s ({name}):  R^2 = {r2:.4f}")

    # Also fit on pooled data (ignoring group)
    probe_all = Ridge(alpha=1.0)
    probe_all.fit(phi_r, phi_s)
    phi_s_hat_all = probe_all.predict(phi_r)
    r2_all = r2_score(phi_s, phi_s_hat_all, multioutput="variance_weighted")
    results["r2_pooled"] = float(r2_all)
    print(f"    Linear map phi_r -> phi_s (pooled):  R^2 = {r2_all:.4f}")

    return results


# ============================================================
#  Null-distribution control (permutation test)
# ============================================================

def permutation_test_linearity(phi_r, phi_s, groups, n_perm=1000, seed=42):
    """
    Permutation test for the linear relationship phi_r -> phi_s.

    For each group g in {0 (majority), 1 (minority)} and for pooled data:
      1. Compute the observed R^2 from Ridge(phi_r -> phi_s).
      2. Repeat n_perm times: shuffle the *rows* of phi_s (as a block,
         preserving the internal correlation structure of phi_s) within
         the group, refit Ridge, record R^2.
      3. Report observed R^2, null mean, null std, and empirical p-value
         (fraction of null R^2 >= observed R^2).

    The permutation is done *within* each group so that group sizes and
    marginal distributions are preserved — we test purely whether the
    sample-to-sample alignment between phi_r and phi_s matters.

    Parameters
    ----------
    phi_r : ndarray (N, n_r)
    phi_s : ndarray (N, n_s)
    groups : ndarray (N,), 0=majority, 1=minority
    n_perm : int, number of permutations
    seed : int

    Returns
    -------
    results : dict with keys like 'majority_observed', 'majority_null_mean',
              'majority_null_std', 'majority_p_value', and similarly for
              'minority' and 'pooled'.
    """
    rng = np.random.RandomState(seed)
    results = {}

    def _fit_r2(X, Y):
        """Ridge R^2 for X -> Y (multi-output, variance-weighted)."""
        probe = Ridge(alpha=1.0)
        probe.fit(X, Y)
        Y_hat = probe.predict(X)
        return r2_score(Y, Y_hat, multioutput="variance_weighted")

    # Per-group + pooled
    configs = [
        ("majority", groups == 0),
        ("minority", groups == 1),
        ("pooled",   np.ones(len(groups), dtype=bool)),
    ]

    for name, mask in configs:
        n_g = mask.sum()
        if n_g < 10:
            print(f"    Permutation test ({name}): skipped (n={n_g})")
            continue

        Xg = phi_r[mask]
        Yg = phi_s[mask]

        observed_r2 = _fit_r2(Xg, Yg)

        null_r2s = np.zeros(n_perm)
        for p in range(n_perm):
            perm_idx = rng.permutation(n_g)
            Yg_perm = Yg[perm_idx]       # shuffle rows as a block
            null_r2s[p] = _fit_r2(Xg, Yg_perm)

        p_value = (null_r2s >= observed_r2).mean()

        results[f"{name}_observed"] = float(observed_r2)
        results[f"{name}_null_mean"] = float(null_r2s.mean())
        results[f"{name}_null_std"] = float(null_r2s.std())
        results[f"{name}_p_value"] = float(p_value)
        results[f"{name}_n_perm"] = n_perm

        print(f"    Permutation test ({name}):  "
              f"R²={observed_r2:.4f}  "
              f"null={null_r2s.mean():.4f}±{null_r2s.std():.4f}  "
              f"p={p_value:.4f}")

    return results


# ============================================================
#  Figures
# ============================================================

def plot_feature_correlations(rho_maj, rho_min, feature_type, out_path):
    """
    Scatter plot: within-majority vs within-minority correlation with
    digit class.  r-features cluster along y=x, s-features along y=-x.
    """
    colors = {"r": "tab:blue", "s": "tab:red",
              "weak": "lightgray", "dead": "black"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for ftype in ["weak", "dead", "s", "r"]:
        idx = [k for k, t in enumerate(feature_type) if t == ftype]
        if idx:
            ax.scatter(rho_maj[idx], rho_min[idx],
                       s=14, alpha=0.6, c=colors[ftype], label=ftype)

    # Reference lines
    lim = max(abs(rho_maj).max(), abs(rho_min).max(), 0.5) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.4,
            label=r"$\rho_0 = \rho_1$ (r)")
    ax.plot([-lim, lim], [lim, -lim], 'k:', lw=0.8, alpha=0.4,
            label=r"$\rho_0 = -\rho_1$ (s)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$\rho_0$ = corr($c_k$, digit | majority)", fontsize=11)
    ax.set_ylabel(r"$\rho_1$ = corr($c_k$, digit | minority)", fontsize=11)
    ax.set_title("SAE feature classification (group-conditional)", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Feature correlation plot saved to {out_path}")


def plot_phi_r_vs_phi_s(c_all, r_idx, s_idx, groups, out_path):
    """
    Scatter plot: first principal direction of phi_r vs first principal
    direction of phi_s, colored by group.
    """
    phi_r = c_all[:, r_idx]
    phi_s = c_all[:, s_idx]

    # Use the first SVD component as a 1D summary
    def first_pc(X):
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return X_centered @ Vt[0]

    pr = first_pc(phi_r)   # (N,)
    ps = first_pc(phi_s)   # (N,)

    fig, ax = plt.subplots(figsize=(7, 6))
    maj = (groups == 0)
    mino = (groups == 1)

    # Subsample for readability
    rng = np.random.RandomState(0)
    max_pts = 3000
    for mask, color, label in [
        (maj, "maroon", "Majority (maroon)"),
        (mino, "teal", "Minority (teal)"),
    ]:
        idx = np.where(mask)[0]
        if len(idx) > max_pts:
            idx = rng.choice(idx, max_pts, replace=False)
        ax.scatter(pr[idx], ps[idx], s=6, alpha=0.3, c=color, label=label)

    ax.set_xlabel(r"First PC of $\varphi_r$ (r-feature activations)", fontsize=11)
    ax.set_ylabel(r"First PC of $\varphi_s$ (s-feature activations)", fontsize=11)
    ax.set_title(r"$\varphi_r$ vs $\varphi_s$ in SAE feature space", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  phi_r vs phi_s scatter saved to {out_path}")


# ============================================================
#  Core processing for one epoch directory
# ============================================================

def process_one(data_dir, out_dir, args, device):
    """Run the full SAE pipeline on one representations.npz."""
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    #  Load representations
    # ------------------------------------------------------------------
    print("[1/6] Loading representations...")
    rep = np.load(os.path.join(data_dir, "representations.npz"))
    phi = rep["phi"]                       # (N, 128)
    groups = rep["groups"]
    digit_classes = rep["digit_classes"]

    N, d_in = phi.shape
    d_hid = d_in * args.dict_ratio
    print(f"  N={N}, d_in={d_in}, dictionary size={d_hid} (R={args.dict_ratio})")

    phi_tensor = torch.from_numpy(phi).float()
    ds = TensorDataset(phi_tensor)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # ------------------------------------------------------------------
    #  Train SAE
    # ------------------------------------------------------------------
    print(f"\n[2/6] Training SAE (alpha={args.alpha}, "
          f"{args.sae_epochs} epochs)...")
    sae = SparseAutoencoder(d_in=d_in, d_hid=d_hid).to(device)
    train_sae(sae, loader, args.sae_epochs, args.sae_lr, args.alpha, device)

    # Save SAE
    sae_path = os.path.join(out_dir, "sae_model.pt")
    torch.save(sae.state_dict(), sae_path)
    print(f"  SAE saved to {sae_path}")

    # ------------------------------------------------------------------
    #  Encode all samples
    # ------------------------------------------------------------------
    print("\n[3/6] Encoding all samples through SAE...")
    sae.eval()
    c_list = []
    with torch.no_grad():
        for i in range(0, N, 2048):
            batch = phi_tensor[i:i+2048].to(device)
            c_batch = sae.encode(batch)
            c_list.append(c_batch.cpu().numpy())
    c_all = np.concatenate(c_list, axis=0)   # (N, d_hid)

    active = (c_all > 0).any(axis=0).sum()
    avg_active = (c_all > 0).sum(axis=1).mean()
    print(f"  Active features: {active}/{d_hid}")
    print(f"  Avg features active per sample: {avg_active:.1f}")

    # Reconstruction quality
    with torch.no_grad():
        phi_hat = sae.decode(torch.from_numpy(c_all).float().to(device))
        phi_hat = phi_hat.cpu().numpy()
    total_var = ((phi - phi.mean(axis=0)) ** 2).sum()
    resid_var = ((phi - phi_hat) ** 2).sum()
    frac_explained = 1 - resid_var / total_var
    print(f"  Variance explained: {frac_explained:.4f}")

    # ------------------------------------------------------------------
    #  Classify features (group-conditional)
    # ------------------------------------------------------------------
    print("\n[4/6] Classifying SAE features (group-conditional)...")
    feature_type, rho_maj, rho_min = classify_features(
        c_all, digit_classes, groups,
        threshold=args.corr_threshold)

    plot_feature_correlations(
        rho_maj, rho_min, feature_type,
        os.path.join(out_dir, "sae_feature_correlations.png"))

    # ------------------------------------------------------------------
    #  Decompose and measure linearity
    # ------------------------------------------------------------------
    print("\n[5/6] Decomposing phi into phi_r and phi_s...")
    results = decompose_and_measure(c_all, feature_type, groups)

    if results.get("r_idx") and results.get("s_idx"):
        plot_phi_r_vs_phi_s(
            c_all, results["r_idx"], results["s_idx"], groups,
            os.path.join(out_dir, "phi_r_vs_phi_s.png"))

    # ------------------------------------------------------------------
    #  Permutation test (null-distribution control)
    # ------------------------------------------------------------------
    null_test_results = {}
    if args.n_perm > 0 and results.get("r_idx") and results.get("s_idx"):
        print(f"\n[6/6] Permutation test ({args.n_perm} permutations)...")
        phi_r = c_all[:, results["r_idx"]]
        phi_s = c_all[:, results["s_idx"]]
        null_test_results = permutation_test_linearity(
            phi_r, phi_s, groups,
            n_perm=args.n_perm, seed=args.seed)
    elif args.n_perm == 0:
        print("\n[6/6] Permutation test: skipped (--n_perm 0)")

    # Save summary
    summary = {
        "sae": {
            "d_in": d_in, "d_hid": d_hid, "dict_ratio": args.dict_ratio,
            "alpha": args.alpha, "epochs": args.sae_epochs,
            "active_features": int(active),
            "avg_active_per_sample": float(avg_active),
            "variance_explained": float(frac_explained),
        },
        "features": {
            "n_r": results["n_r"], "n_s": results["n_s"],
            "n_dead": results.get("n_dead", 0),
            "n_weak": results.get("n_weak", 0),
        },
        "linearity": {
            "r2_majority": results.get("r2_majority"),
            "r2_minority": results.get("r2_minority"),
            "r2_pooled": results.get("r2_pooled"),
        },
        "corr_threshold": args.corr_threshold,
    }
    if null_test_results:
        summary["null_test"] = null_test_results

    summary_path = os.path.join(out_dir, "sae_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")
    return summary


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train SAE on MLP representations and analyze "
                    "spurious correlation preservation.")
    parser.add_argument("--data_dir", type=str, default="./data_colored_mnist",
                        help="Directory with representations.npz, or parent "
                             "of epoch_* subdirs when using --run_all_epochs")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: same as data_dir)")
    parser.add_argument("--run_all_epochs", action="store_true",
                        help="Loop over all epoch_* subdirs in data_dir")
    # SAE hyperparameters
    parser.add_argument("--dict_ratio", type=int, default=4,
                        help="Ratio of dictionary size to input dim (R)")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="L1 sparsity coefficient")
    parser.add_argument("--sae_epochs", type=int, default=30)
    parser.add_argument("--sae_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    # Feature classification
    parser.add_argument("--corr_threshold", type=float, default=0.2,
                        help="Min |correlation| to classify a feature as r or s")
    parser.add_argument("--seed", type=int, default=42)
    # Permutation test
    parser.add_argument("--n_perm", type=int, default=0,
                        help="Number of permutations for null-distribution "
                             "control (0 = skip, e.g. 1000)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.run_all_epochs:
        # Find all epoch_* subdirs and process each
        pattern = os.path.join(args.data_dir, "epoch_*")
        epoch_dirs = sorted(glob.glob(pattern))
        if not epoch_dirs:
            print(f"No epoch_* directories found in {args.data_dir}")
            return
        print(f"Found {len(epoch_dirs)} epoch directories: "
              f"{[os.path.basename(d) for d in epoch_dirs]}\n")

        all_summaries = {}
        for edir in epoch_dirs:
            tag = os.path.basename(edir)
            out = edir if args.out_dir is None else os.path.join(args.out_dir, tag)
            print(f"\n{'='*60}")
            print(f"  Processing {tag}")
            print(f"{'='*60}")
            summary = process_one(edir, out, args, device)
            all_summaries[tag] = summary

        # Save combined summary
        combined_path = os.path.join(args.data_dir, "sae_all_epochs_summary.json")
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nCombined summary saved to {combined_path}")
    else:
        data_dir = args.data_dir
        out_dir = args.out_dir if args.out_dir else data_dir
        process_one(data_dir, out_dir, args, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
