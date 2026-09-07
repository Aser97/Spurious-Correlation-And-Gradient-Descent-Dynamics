"""
synth_utils.py
Utilities for synthetic isotropic-regime experiments.

Data model (matches the paper exactly):
  - r ~ compactly supported distribution in R^{d_r}, with ||v||=1
  - On G_maj (prob 1-eps):  s = A r + xi
  - On G_min (prob eps):    s = B r + xi
  - xi ~ symmetric, bounded, independent of (y, r)
  - y absorbed into x (i.e. x <- y*x), so y=+1 always.

Isotropic regime: v is a common eigenvector of A^T A, A^T B, B^T B, B^T A
  with eigenvalues mu_A, mu, mu_B, mu respectively.

We construct A, B explicitly to satisfy these spectral conditions.
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
from typing import Optional


# ============================================================
#  Configuration
# ============================================================

@dataclass
class SynthConfig:
    """All knobs for one synthetic experiment run."""
    # Geometry
    d_r: int = 8               # dimension of r
    d_s: int = 8               # dimension of s  (must be >= d_r for isometries)
    gamma_min: float = 1.0     # tilde-gamma_min  (r-margin of minority)
    gamma_maj: float = 1.0     # tilde-gamma_min  (r-margin of majority, kept =1 WLOG)
    mu_A: float = 1.0          # eigenvalue v^T A^T A v
    mu_B: float = 1.0          # eigenvalue v^T B^T B v
    mu: float = -0.5           # eigenvalue v^T A^T B v  (coupling)
    noise_R: float = 0.0       # ||xi|| <= noise_R  (0 = noiseless)

    # Training
    epsilon: float = 0.1       # minority fraction
    N: int = 50_000            # dataset size  (large = population-level approx)
    lr: float = 0.01           # constant learning rate h
    steps: int = 500_000       # total gradient steps T
    seed: int = 42

    # Logging
    log_every: int = 500
    print_every: int = 50_000
    out_dir: str = "./runs_synth/default"

    # Derived (computed in __post_init__)
    alpha: float = field(init=False)

    def __post_init__(self):
        self.alpha = self.gamma_min * (1 + self.mu) / (1 + self.mu_A)

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
#  Constructing A, B in the isotropic regime
# ============================================================

def build_isotropic_operators(cfg: SynthConfig):
    """
    Construct A, B in R^{d_s x d_r} such that:
      v^T A^T A v = mu_A,   v^T A^T B v = mu,
      v^T B^T B v = mu_B,   v^T B^T A v = mu.

    Strategy:
      Choose v = e_1 (first standard basis vector in R^{d_r}).
      Let A = U_A @ D_A  where U_A is d_s x d_r with orthonormal columns
      and D_A is diagonal d_r x d_r.

      For simplicity, we use:
        A v = sqrt(mu_A) * u_1        (u_1 = first col of U_A)
        B v = (mu / sqrt(mu_A)) * u_1 + sqrt(mu_B - mu^2/mu_A) * u_2

      This gives:
        v^T A^T A v = mu_A            check
        v^T A^T B v = mu              check
        v^T B^T B v = mu_B            check  (since mu^2/mu_A + mu_B - mu^2/mu_A = mu_B)
        v^T B^T A v = mu              check  (transpose of A^T B restricted to v)

      For directions perpendicular to v, A and B act as identity (simplest choice).
    """
    d_r, d_s = cfg.d_r, cfg.d_s
    assert d_s >= d_r, "Need d_s >= d_r"
    assert cfg.mu_A > 0 and cfg.mu_B > 0
    assert cfg.mu**2 <= cfg.mu_A * cfg.mu_B, "Need mu^2 <= mu_A * mu_B for PSD"
    assert 1 + cfg.mu >= 0, "Need 1 + mu >= 0 (attractive regime)"

    # v = e_1 in R^{d_r}
    v = np.zeros(d_r)
    v[0] = 1.0

    # Build A as d_s x d_r matrix
    A = np.zeros((d_s, d_r))
    # Column 0 of A (= A v):  sqrt(mu_A) * e_1 in R^{d_s}
    A[0, 0] = math.sqrt(cfg.mu_A)
    # Remaining columns: identity mapping (r_perp -> s_perp)
    for j in range(1, d_r):
        A[j, j] = 1.0  # simple embedding

    # Build B as d_s x d_r matrix
    B = np.zeros((d_s, d_r))
    # Column 0 of B (= B v): decompose into u_1 and u_2 components
    c1 = cfg.mu / math.sqrt(cfg.mu_A)
    c2_sq = cfg.mu_B - cfg.mu**2 / cfg.mu_A
    c2 = math.sqrt(max(c2_sq, 0.0))  # might be zero if mu^2 = mu_A mu_B
    B[0, 0] = c1   # component along u_1 = e_1 in R^{d_s}
    B[1, 0] = c2   # component along u_2 = e_2 in R^{d_s}
    # Remaining columns: same as A
    for j in range(1, d_r):
        B[j, j] = 1.0

    # Verify
    AtA_v = A.T @ A @ v
    AtB_v = A.T @ B @ v
    BtB_v = B.T @ B @ v
    BtA_v = B.T @ A @ v
    assert abs(v @ AtA_v - cfg.mu_A) < 1e-10, f"mu_A check failed: {v @ AtA_v}"
    assert abs(v @ AtB_v - cfg.mu)   < 1e-10, f"mu check failed: {v @ AtB_v}"
    assert abs(v @ BtB_v - cfg.mu_B) < 1e-10, f"mu_B check failed: {v @ BtB_v}"
    assert abs(v @ BtA_v - cfg.mu)   < 1e-10, f"mu (transpose) check failed: {v @ BtA_v}"

    A_t = torch.tensor(A, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)
    v_t = torch.tensor(v, dtype=torch.float32)
    return A_t, B_t, v_t


# ============================================================
#  Synthetic dataset generation
# ============================================================

def generate_dataset(cfg: SynthConfig, A: torch.Tensor, B: torch.Tensor, v: torch.Tensor):
    """
    Generate the label-absorbed dataset.

    Following the paper's convention, we absorb the label into the features:
    x <- y * x, so that y = +1 for all samples.  In this representation,
    the margin condition becomes  hat_w . x >= 1  (always positive).

    Concretely (all in the absorbed space):
      - Z = v^T r  drawn from Uniform[gamma_g, K]  (always positive)
      - r_perp ~ uniform on a ball of radius R_perp in Span(v)^perp
      - r = Z * v + r_perp
      - On G_maj:  s = A r  (+ noise xi if enabled)
      - On G_min:  s = B r  (+ noise xi if enabled)
      - x = [r; s]

    Since labels are absorbed, the logistic loss is  log(1 + exp(-w^T x))
    and the gradient step is  w <- w + h * (1/N) sum_i sigma(-w^T x_i) * x_i.

    Returns: X (N, d_r+d_s), groups (N,), metadata dict
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device
    N = cfg.N
    d_r, d_s = cfg.d_r, cfg.d_s
    K = max(cfg.gamma_maj, cfg.gamma_min) + 2.0  # upper bound of support
    R_perp = 0.5  # radius of r_perp ball

    # Assign groups:  0 = majority, 1 = minority
    groups = (torch.rand(N) < cfg.epsilon).long()  # 1 = minority
    n_min = groups.sum().item()
    n_maj = N - n_min

    # Draw Z = v^T r  from Uniform[gamma_g, K]   (positive, label already absorbed)
    gamma_g = torch.where(groups == 0, cfg.gamma_maj, cfg.gamma_min).float()
    Z = gamma_g + torch.rand(N) * (K - gamma_g)  # Uniform[gamma_g, K]

    # Draw r_perp ~ uniform on ball in R^{d_r - 1}, embed in R^{d_r}
    r_perp_raw = torch.randn(N, d_r - 1)
    r_perp_raw = r_perp_raw / (r_perp_raw.norm(dim=1, keepdim=True) + 1e-12)
    r_perp_scale = R_perp * torch.rand(N, 1).pow(1.0 / max(d_r - 1, 1))
    r_perp_raw = r_perp_raw * r_perp_scale

    # Build r = Z * v + r_perp  (v = e_1, so r_perp lives in dims 1..d_r-1)
    r = torch.zeros(N, d_r)
    r[:, 0] = Z
    r[:, 1:] = r_perp_raw

    # Build s = A r (majority) or B r (minority)
    A_np, B_np = A.cpu(), B.cpu()
    s = torch.zeros(N, d_s)
    maj_mask = (groups == 0)
    min_mask = (groups == 1)
    s[maj_mask] = (r[maj_mask] @ A_np.T)
    s[min_mask] = (r[min_mask] @ B_np.T)

    # Add noise xi if requested
    if cfg.noise_R > 0:
        xi = torch.randn(N, d_s)
        xi = xi / (xi.norm(dim=1, keepdim=True) + 1e-12)
        xi = xi * cfg.noise_R * torch.rand(N, 1).pow(1.0 / d_s)
        s = s + xi

    # Full feature vector (already label-absorbed)
    x = torch.cat([r, s], dim=1)  # (N, d_r + d_s)

    # Move to device
    X = x.to(device)
    groups = groups.to(device)

    meta = {
        "N": N, "n_maj": n_maj, "n_min": n_min,
        "d_r": d_r, "d_s": d_s,
        "K": K, "R_perp": R_perp,
        "gamma_maj": cfg.gamma_maj, "gamma_min": cfg.gamma_min,
        "mu_A": cfg.mu_A, "mu_B": cfg.mu_B, "mu": cfg.mu,
        "alpha": cfg.alpha, "epsilon": cfg.epsilon,
    }
    return X, groups, meta


# ============================================================
#  Training loop
# ============================================================

def train_population_gd(cfg: SynthConfig, X: torch.Tensor, groups: torch.Tensor):
    """
    Full-batch gradient descent with logistic loss on the label-absorbed data.
    Since y is absorbed, the logit is w^T x and the loss is log(1 + exp(-w^T x)).

    Returns: logs dict with trajectories.
    """
    device = cfg.device
    N, d = X.shape
    w = torch.zeros(d, device=device, requires_grad=False)

    maj_mask = (groups == 0)
    min_mask = (groups == 1)
    n_maj = maj_mask.sum().float()
    n_min = min_mask.sum().float()

    logs = {
        "t": [], "err_maj": [], "err_min": [],
        "err_maj_rescaled": [], "err_min_rescaled": [],
        "loss": [],
    }

    for t in range(1, cfg.steps + 1):
        # Forward: logit = X @ w,  loss = mean log(1 + exp(-logit))
        logit = X @ w                          # (N,)
        sigmoid_pos = torch.sigmoid(logit)     # p_y = sigma(w^T x)
        q = 1.0 - sigmoid_pos                  # 1 - p_y

        # Gradient:  -1/N * sum_i (1-p_y_i) * x_i  =  -1/N * X^T (1 - sigmoid)
        # But we want to MINIMIZE, so update is w <- w + lr * (1/N) X^T q
        grad = X.T @ q / N                     # (d,)
        w = w + cfg.lr * grad

        # Logging
        if (t % cfg.log_every == 0) or t == 1:
            with torch.no_grad():
                err_maj = q[maj_mask].mean().item() if n_maj > 0 else float('nan')
                err_min = q[min_mask].mean().item() if n_min > 0 else float('nan')
                loss_val = torch.log1p(torch.exp(-logit)).mean().item()

                z_t = cfg.lr * t  # cumulative learning = h * t for constant h
                eps = cfg.epsilon

                # Rescaled: err * eps_g * z_t  (should -> kappa_g if theory holds)
                err_maj_resc = err_maj * (1 - eps) * z_t
                err_min_resc = err_min * eps * z_t

                logs["t"].append(t)
                logs["err_maj"].append(err_maj)
                logs["err_min"].append(err_min)
                logs["err_maj_rescaled"].append(err_maj_resc)
                logs["err_min_rescaled"].append(err_min_resc)
                logs["loss"].append(loss_val)

            if (t % cfg.print_every == 0) or t == 1:
                print(f"  [t={t:>7d}]  loss={loss_val:.6f}  "
                      f"err_maj={err_maj:.6f}  err_min={err_min:.6f}  "
                      f"resc_maj={err_maj_resc:.4f}  resc_min={err_min_resc:.4f}")

    return logs, w


# ============================================================
#  Theory predictions (closed-form kappa)
# ============================================================

def kappa_theory(cfg: SynthConfig):
    """
    Compute the closed-form prefactors from Theorem 2 (isotropic regime).

    When alpha < 1:
      kappa_maj = [gamma_min*(1+mu_B) - (1+mu)] / [gamma_min * Sigma]
      kappa_min = [(1+mu_A) - gamma_min*(1+mu)] / [gamma_min^2 * Sigma]
      where Sigma = (1+mu_A)(1+mu_B) - (1+mu)^2

    When alpha >= 1:
      kappa_maj = 1 / (1 + mu_A)
      kappa_min = None (rate is z_t^{-alpha}, not 1/z_t)
    """
    gm = cfg.gamma_min
    ma, mb, m = cfg.mu_A, cfg.mu_B, cfg.mu
    Sigma = (1 + ma) * (1 + mb) - (1 + m)**2

    if cfg.alpha < 1.0:
        km = (gm * (1 + mb) - (1 + m)) / (gm * Sigma)
        kn = ((1 + ma) - gm * (1 + m)) / (gm**2 * Sigma)
        return {"kappa_maj": km, "kappa_min": kn, "Sigma": Sigma, "alpha": cfg.alpha}
    else:
        km = 1.0 / (1 + ma)
        return {"kappa_maj": km, "kappa_min": None, "Sigma": Sigma, "alpha": cfg.alpha}


# ============================================================
#  I/O helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_logs(logs, meta, cfg):
    ensure_dir(cfg.out_dir)
    payload = {"config": {k: v for k, v in asdict(cfg).items()}, "meta": meta, "logs": logs}
    path = os.path.join(cfg.out_dir, "logs.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_logs(run_dir):
    path = os.path.join(run_dir, "logs.json")
    with open(path) as f:
        return json.load(f)
