#!/usr/bin/env python3
"""
run_tabular_vendor.py
=====================
Setup 1: Tabular Data with Third-Party Vendor Features

Narrative
---------
In many real-world tabular pipelines (credit scoring, medical diagnostics),
practitioners concatenate raw, verified measurements with derived "risk indices"
provided by external vendors:

  x = [r ; s]

where r in R^{d_r} are the raw causal features and s in R^{d_s} are the vendor
risk scores.  Two different geographic regions (groups) use different vendors,
whose proprietary aggregation matrices A and B map r to s differently:

  Group 0 (majority): s = A * r + xi
  Group 1 (minority): s = B * r + xi

Both vendors attempt to predict the same underlying phenomenon, so their matrices
share the same singular vector basis but differ in singular values.  This yields
the isotropic regime by construction: v is a common eigenvector of A^T A, A^T B,
B^T B, B^T A.

Under IRM, the vendor score is spurious because its optimal linear weight changes
across groups (A != B).  A model that relies on it fails to generalise.

Experiment design
-----------------
  - Sweep alpha in {0.5, 0.7, 1.0, 1.3, 1.5, 2.0}
  - For alpha < 1: both groups decay as kappa_g / (eps_g * z_t)
  - For alpha >= 1: minority escapes eps-dependence, decays as z_t^{-alpha}
  - Discriminating visualisations: error ratio, compensated errors, rescaled errors

Usage
-----
  # Full run (default: 10M steps, N=5M, ~several hours on GPU):
  python run_tabular_vendor.py

  # Custom:
  python run_tabular_vendor.py --N 100000 --out_dir ./tabular_results

  # Quick sanity check (~5 min):
  python run_tabular_vendor.py --quick

  # Re-plot only (no GD re-run), optionally cropping x-axis:
  python run_tabular_vendor.py --plot_only
  python run_tabular_vendor.py --plot_only --z_max 20000
"""

import argparse
import os
import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from synth_utils import (
    SynthConfig, build_isotropic_operators, generate_dataset,
    train_population_gd, kappa_theory, save_logs, load_logs, ensure_dir,
)


# ============================================================
#  Parameter configurations for the alpha sweep
# ============================================================

def make_alpha_configs(alphas, epsilon, steps, N, noise_R, out_root):
    """
    For each target alpha, choose (mu_A, mu_B, mu, gamma_min) that yield it.

    Strategy: fix mu_A = mu_B = 1.0, mu = 0.0, and vary gamma_min.
      alpha = gamma_min * (1 + mu) / (1 + mu_A) = gamma_min / 2

    So gamma_min = 2 * alpha.  This isolates the margin advantage as the
    sole driver of the phase transition, matching the paper's cleanest case.
    """
    configs = []
    for a in alphas:
        # Build config with correct parameters from the start so that
        # __post_init__ computes alpha correctly.
        cfg = SynthConfig(
            mu_A=1.0,
            mu_B=1.0,
            mu=0.0,
            gamma_min=2.0 * a,   # => alpha = gamma_min * 1 / 2 = a
            gamma_maj=1.0,
            epsilon=epsilon,
            noise_R=noise_R,
            N=N,
            steps=steps,
            lr=0.01,
            log_every=max(1, steps // 2000),   # ~2000 log points
            print_every=steps // 10,
            out_dir=os.path.join(out_root, f"alpha_{a:.2f}"),
        )
        assert abs(cfg.alpha - a) < 1e-10, \
            f"Alpha mismatch: expected {a}, got {cfg.alpha}"
        configs.append(cfg)
    return configs


# ============================================================
#  Discriminating visualisations
# ============================================================

def smooth(arr, window=51):
    """Simple moving average for noisy curves."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def make_discriminating_plots(logs, cfg, theory, out_dir, z_max=None):
    """
    Produce 3 plots per alpha value, tailored to the regime:

      1. Error ratio:  err_min / err_maj
           alpha < 1  => converges to constant
           alpha >= 1 => diverges (minority decays faster)

      2. Compensated errors:  err * z_t^beta
           alpha < 1:  single panel, err*z_t (both plateau)
           alpha = 1:  single panel, err*z_t (majority plateaus, minority -> 0)
           alpha > 1:  two panels: err*z_t + minority-only err_min*z_t^alpha

      3. Rescaled errors:  eps_g * err_g * z_t  (shows kappa_g prefactors)

    If z_max is given, crop all plots at z_t <= z_max (to remove
    the finite-sample floor region where errors can't decrease further).
    """
    ensure_dir(out_dir)
    t = np.array(logs["t"], dtype=np.float64)
    err_maj = np.array(logs["err_maj"])
    err_min = np.array(logs["err_min"])
    z_t = cfg.lr * t
    alpha = cfg.alpha

    # Filter out early transient and numerically noisy tail
    # Errors below ~1e-7 are dominated by floating-point noise in finite sums,
    # making log-derivatives unreliable.
    ERR_FLOOR = 1e-7
    mask = (err_maj > ERR_FLOOR) & (err_min > ERR_FLOOR) & (z_t > 1.0)
    if z_max is not None:
        mask &= (z_t <= z_max)
    t, z_t, err_maj, err_min = t[mask], z_t[mask], err_maj[mask], err_min[mask]

    if len(t) < 100:
        print(f"  [WARNING] Not enough data points for plots (alpha={alpha:.2f})")
        return

    # --- Plot 1: Error ratio ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ratio = err_min / err_maj
    ax.plot(z_t, ratio, "k-", lw=1.5)
    ax.set_xlabel("$z_t = h \\cdot t$")
    ax.set_ylabel("$\\mathrm{err}_{\\min} \\,/\\, \\mathrm{err}_{\\mathrm{maj}}$")
    ax.set_xscale("log")
    if alpha < 1.0:
        # Should converge to (kappa_min / eps_min) / (kappa_maj / eps_maj)
        # = (kappa_min * eps_maj) / (kappa_maj * eps_min)
        if theory.get("kappa_min") is not None:
            expected = (theory["kappa_min"] * (1 - cfg.epsilon)) / \
                       (theory["kappa_maj"] * cfg.epsilon)
            ax.axhline(expected, color="green", ls="--", lw=1.5,
                       label=f"Theory: {expected:.2f}")
            ax.legend()
        ax.set_title(f"Error ratio (α = {alpha:.2f} < 1: should converge)")
    else:
        ax.set_yscale("log")
        ax.set_ylabel("$\\mathrm{err}_{\\min} \\,/\\, \\mathrm{err}_{\\mathrm{maj}}$ (log)")
        ax.set_title(f"Error ratio (α = {alpha:.2f} ≥ 1: should diverge)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_ratio.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "error_ratio.pdf"))
    plt.close(fig)

    # --- Plot 3: Compensated errors ---
    # Layout depends on the regime:
    #   alpha < 1:  single panel, err*z_t (both groups should plateau)
    #   alpha = 1:  single panel, err*z_t (majority plateaus, minority -> 0 slowly)
    #   alpha > 1:  two panels: err*z_t and err_min*z_t^alpha (minority only,
    #               since majority*z_t^alpha diverges and hides the plateau)
    if alpha > 1.0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: err * z_t  =>  majority plateaus, minority -> 0
        axes[0].plot(z_t, err_maj * z_t, "b-", lw=1.5, label="Majority · $z_t$")
        axes[0].plot(z_t, err_min * z_t, "r-", lw=1.5, label="Minority · $z_t$")
        if theory.get("kappa_maj") is not None:
            axes[0].axhline(theory["kappa_maj"] / (1 - cfg.epsilon), color="blue",
                            ls="--", alpha=0.6,
                            label=f"Theory: κ_maj/(1−ε) = {theory['kappa_maj']/(1-cfg.epsilon):.3f}")
        axes[0].set_xlabel("$z_t$")
        axes[0].set_ylabel("$\\mathrm{err} \\cdot z_t$")
        axes[0].set_title("Compensated by $z_t$\n(plateau ↔ $1/z_t$ decay)")
        axes[0].set_xscale("log")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Right: err_min * z_t^alpha only (majority diverges, would hide plateau)
        axes[1].plot(z_t, err_min * z_t**alpha, "r-", lw=1.5,
                     label=f"Minority · $z_t^{{{alpha:.2f}}}$")
        axes[1].set_xlabel("$z_t$")
        axes[1].set_ylabel(f"$\\mathrm{{err}}_{{\\min}} \\cdot z_t^{{{alpha:.2f}}}$")
        axes[1].set_title(f"Minority compensated by $z_t^{{\\alpha}}$\n"
                          f"(plateau ↔ $z_t^{{-\\alpha}}$ decay)")
        axes[1].set_xscale("log")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Compensated errors  (α = {alpha:.2f})", fontsize=13)
    else:
        # Single panel: err * z_t for both groups
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(z_t, err_maj * z_t, "b-", lw=1.5, label="Majority · $z_t$")
        ax.plot(z_t, err_min * z_t, "r-", lw=1.5, label="Minority · $z_t$")
        if theory.get("kappa_maj") is not None:
            ax.axhline(theory["kappa_maj"] / (1 - cfg.epsilon), color="blue",
                       ls="--", alpha=0.6,
                       label=f"Theory: κ_maj/(1−ε) = {theory['kappa_maj']/(1-cfg.epsilon):.3f}")
        if theory.get("kappa_min") is not None:
            ax.axhline(theory["kappa_min"] / cfg.epsilon, color="red",
                       ls="--", alpha=0.6,
                       label=f"Theory: κ_min/ε = {theory['kappa_min']/cfg.epsilon:.3f}")
        ax.set_xlabel("$z_t$")
        ax.set_ylabel("$\\mathrm{err} \\cdot z_t$")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if alpha < 1.0:
            ax.set_title(f"Compensated errors (α = {alpha:.2f} < 1: both groups plateau)")
        else:
            ax.set_title(f"Compensated errors (α = 1.0: majority plateaus, "
                         f"minority → 0 with log correction)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compensated.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "compensated.pdf"))
    plt.close(fig)

    # --- Plot 4: Rescaled errors (auto-scaled) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    rescaled_maj = (1 - cfg.epsilon) * err_maj * z_t
    rescaled_min = cfg.epsilon * err_min * z_t
    ax.plot(z_t, rescaled_maj, "b-", lw=1.5, label="$(1-\\varepsilon)\\cdot\\mathrm{err}_{\\mathrm{maj}}\\cdot z_t$")
    ax.plot(z_t, rescaled_min, "r-", lw=1.5, label="$\\varepsilon\\cdot\\mathrm{err}_{\\min}\\cdot z_t$")
    if theory.get("kappa_maj") is not None:
        ax.axhline(theory["kappa_maj"], color="blue", ls="--", alpha=0.6,
                   label=f"κ_maj = {theory['kappa_maj']:.4f}")
    if theory.get("kappa_min") is not None:
        ax.axhline(theory["kappa_min"], color="red", ls="--", alpha=0.6,
                   label=f"κ_min = {theory['kappa_min']:.4f}")
    ax.set_xlabel("$z_t = h \\cdot t$")
    ax.set_ylabel("Rescaled error")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if alpha < 1.0:
        ax.set_title(f"Rescaled errors (α = {alpha:.2f} < 1: both plateau at κ_g)")
    elif alpha == 1.0:
        ax.set_title(f"Rescaled errors (α = 1.0: majority plateaus, minority → 0)")
    else:
        ax.set_title(f"Rescaled errors (α = {alpha:.2f} > 1: majority plateaus, minority → 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rescaled_errors.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "rescaled_errors.pdf"))
    plt.close(fig)


def make_summary_plot(all_results, out_dir, z_max=None):
    """
    Summary plot: measured minority decay exponent vs. theoretical alpha.
    Measures the exponent from the last 20% of the local slope curve.
    """
    ensure_dir(out_dir)
    alphas_theory = []
    exponents_measured_min = []
    exponents_measured_maj = []

    for res in all_results:
        cfg = res["cfg"]
        logs = res["logs"]
        t = np.array(logs["t"], dtype=np.float64)
        err_maj = np.array(logs["err_maj"])
        err_min = np.array(logs["err_min"])
        z_t = cfg.lr * t

        mask = (err_maj > 1e-7) & (err_min > 1e-7) & (z_t > 1.0)
        if z_max is not None:
            mask &= (z_t <= z_max)
        z_t, err_maj, err_min = z_t[mask], err_maj[mask], err_min[mask]

        if len(z_t) < 50:
            continue

        log_z = np.log(z_t)
        dlog_z = np.diff(log_z)

        # Majority exponent (last 20%)
        slope_maj = np.diff(np.log(err_maj)) / dlog_z
        n_tail = max(10, len(slope_maj) // 5)
        exp_maj = np.median(slope_maj[-n_tail:])

        # Minority exponent (last 20%)
        slope_min = np.diff(np.log(err_min)) / dlog_z
        exp_min = np.median(slope_min[-n_tail:])

        alphas_theory.append(cfg.alpha)
        exponents_measured_maj.append(-exp_maj)
        exponents_measured_min.append(-exp_min)

    alphas_theory = np.array(alphas_theory)
    exponents_measured_min = np.array(exponents_measured_min)
    exponents_measured_maj = np.array(exponents_measured_maj)

    fig, ax = plt.subplots(figsize=(8, 6))
    a_range = np.linspace(0.3, 2.2, 100)
    # Theory: majority always decays as z_t^{-1}
    ax.plot(a_range, np.ones_like(a_range), "b--", lw=1.5, alpha=0.6,
            label="Theory: majority exponent = 1")
    # Theory: minority decays as z_t^{-min(alpha, 1)} ... actually:
    #   alpha < 1: minority decays as 1/z_t => exponent = 1
    #   alpha >= 1: minority decays as z_t^{-alpha} => exponent = alpha
    theory_min_exp = np.where(a_range < 1, 1.0, a_range)
    ax.plot(a_range, theory_min_exp, "r--", lw=1.5, alpha=0.6,
            label="Theory: minority exponent = max(α, 1)")

    ax.scatter(alphas_theory, exponents_measured_maj, c="blue", s=80, zorder=5,
               marker="o", label="Measured: majority")
    ax.scatter(alphas_theory, exponents_measured_min, c="red", s=80, zorder=5,
               marker="^", label="Measured: minority")

    ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.text(1.02, 0.5, "α = 1\n(phase transition)", fontsize=9, color="gray",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("Theoretical α")
    ax.set_ylabel("Measured decay exponent (−slope)")
    ax.set_title("Phase transition verification: measured exponent vs. α")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.3, 2.2])
    ax.set_ylim([0.0, 2.5])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_transition_summary.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "phase_transition_summary.pdf"))
    plt.close(fig)
    print(f"  Summary plot saved to {out_dir}/phase_transition_summary.png")


# ============================================================
#  Main pipeline
# ============================================================

def run_single(cfg):
    """Run GD for one configuration and return logs."""
    print(f"\n{'='*60}")
    print(f"  alpha = {cfg.alpha:.3f}  (gamma_min={cfg.gamma_min:.2f}, "
          f"mu_A={cfg.mu_A}, mu_B={cfg.mu_B}, mu={cfg.mu})")
    print(f"  epsilon = {cfg.epsilon}, N = {cfg.N}, steps = {cfg.steps}")
    print(f"{'='*60}")

    A, B, v = build_isotropic_operators(cfg)
    X, groups, meta = generate_dataset(cfg, A, B, v)

    print(f"  Dataset: N={meta['N']}, n_maj={meta['n_maj']}, n_min={meta['n_min']}")
    print(f"  Running GD ({cfg.steps} steps, lr={cfg.lr})...")

    logs, w_final = train_population_gd(cfg, X, groups)

    # Save logs
    save_logs(logs, meta, cfg)

    # Compute theory predictions
    theory = kappa_theory(cfg)
    print(f"  Theory: alpha={theory['alpha']:.3f}, kappa_maj={theory['kappa_maj']:.4f}, "
          f"kappa_min={theory.get('kappa_min', 'N/A')}")

    # Final empirical errors
    print(f"  Final: err_maj={logs['err_maj'][-1]:.2e}, err_min={logs['err_min'][-1]:.2e}")

    return logs, theory


def main():
    parser = argparse.ArgumentParser(description="Setup 1: Tabular + Vendor Scores")
    parser.add_argument("--out_dir", default="./tabular_vendor_results",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=10_000_000,
                        help="GD steps per configuration")
    parser.add_argument("--N", type=int, default=5_000_000,
                        help="Dataset size")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Minority fraction")
    parser.add_argument("--noise_R", type=float, default=0.1,
                        help="Vendor noise bound ||xi|| <= noise_R")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.5, 0.7, 1.0, 1.3, 1.5, 2.0],
                        help="Alpha values to sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check (fewer steps, fewer alphas)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Re-generate plots from saved logs.json without re-running GD")
    parser.add_argument("--z_max", type=float, default=None,
                        help="Crop x-axis at this z_t value (useful to remove floor region)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.steps = 100_000
        args.N = 10_000
        args.alphas = [0.5, 1.0, 1.5]

    print("=" * 60)
    print("  Setup 1: Tabular Data + Third-Party Vendor Features")
    print("=" * 60)
    print(f"  Output: {args.out_dir}")
    print(f"  Alphas: {args.alphas}")
    if args.plot_only:
        print(f"  Mode: PLOT ONLY (re-generating plots from saved logs)")
    else:
        print(f"  Steps: {args.steps}, N: {args.N}, eps: {args.epsilon}")
        print(f"  Noise: {args.noise_R}, Seed: {args.seed}")
    if args.z_max is not None:
        print(f"  z_max crop: {args.z_max:.0f}")

    # ---- Plot-only mode: reload from saved logs.json ----
    if args.plot_only:
        all_results = []
        for a in args.alphas:
            run_dir = os.path.join(args.out_dir, f"alpha_{a:.2f}")
            logs_path = os.path.join(run_dir, "logs.json")
            if not os.path.exists(logs_path):
                print(f"  [SKIP] No logs.json found for alpha={a:.2f} in {run_dir}")
                continue
            payload = load_logs(run_dir)
            logs = payload["logs"]
            cfg_dict = payload["config"]
            # Reconstruct SynthConfig
            cfg = SynthConfig(
                mu_A=cfg_dict["mu_A"], mu_B=cfg_dict["mu_B"], mu=cfg_dict["mu"],
                gamma_min=cfg_dict["gamma_min"], gamma_maj=cfg_dict["gamma_maj"],
                epsilon=cfg_dict["epsilon"], noise_R=cfg_dict.get("noise_R", 0.1),
                N=cfg_dict["N"], steps=cfg_dict["steps"], lr=cfg_dict["lr"],
                log_every=cfg_dict.get("log_every", 1),
                print_every=cfg_dict.get("print_every", 1),
                out_dir=run_dir,
            )
            theory = kappa_theory(cfg)
            all_results.append({"cfg": cfg, "logs": logs, "theory": theory})
            make_discriminating_plots(logs, cfg, theory, run_dir, z_max=args.z_max)
            print(f"  Plots regenerated for alpha={a:.2f} in {run_dir}/")

        if all_results:
            make_summary_plot(all_results, args.out_dir, z_max=args.z_max)
        print("\nDone (plot-only mode).")
        return

    # ---- Normal mode: run GD then plot ----
    configs = make_alpha_configs(
        alphas=args.alphas,
        epsilon=args.epsilon,
        steps=args.steps,
        N=args.N,
        noise_R=args.noise_R,
        out_root=args.out_dir,
    )
    # Override seed
    for cfg in configs:
        cfg.seed = args.seed

    all_results = []
    for cfg in configs:
        logs, theory = run_single(cfg)
        all_results.append({"cfg": cfg, "logs": logs, "theory": theory})

        # Produce per-alpha plots
        make_discriminating_plots(logs, cfg, theory, cfg.out_dir, z_max=args.z_max)
        print(f"  Plots saved to {cfg.out_dir}/")

    # Summary plot
    make_summary_plot(all_results, args.out_dir, z_max=args.z_max)

    # Save summary JSON
    summary = {}
    for res in all_results:
        cfg = res["cfg"]
        theory = res["theory"]
        logs = res["logs"]
        summary[f"alpha_{cfg.alpha:.2f}"] = {
            "alpha": cfg.alpha,
            "gamma_min": cfg.gamma_min,
            "epsilon": cfg.epsilon,
            "kappa_maj": theory["kappa_maj"],
            "kappa_min": theory.get("kappa_min"),
            "Sigma": theory["Sigma"],
            "final_err_maj": logs["err_maj"][-1],
            "final_err_min": logs["err_min"][-1],
        }
    summary_path = os.path.join(args.out_dir, "summary.json")
    ensure_dir(args.out_dir)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON saved to {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
