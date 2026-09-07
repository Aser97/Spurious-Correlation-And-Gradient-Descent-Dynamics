#!/usr/bin/env python3
"""
run_tabular_vendor_eps_sweep.py
===============================
Epsilon-sweep experiment using the Tabular + Vendor Score setup.

For two fixed alpha values (one < 1, one > 1), sweep epsilon to validate:
  - alpha < 1: minority error ~ kappa_min / (eps * z_t), so raw curves fan out
    by 1/eps, but rescaled errors (eps * err_min * z_t) collapse onto kappa_min.
  - alpha >= 1: minority error ~ z_t^{-alpha} INDEPENDENT of eps. All minority
    curves collapse; majority curves fan out as 1/(1-eps).

Uses the same vendor data generator as run_tabular_vendor.py (isotropic regime
by construction: mu_A = mu_B = 1.0, mu = 0.0, gamma_min = 2*alpha).

Usage
-----
  # Full run (default: 10M steps, N=1M):
  python run_tabular_vendor_eps_sweep.py

  # Quick sanity check:
  python run_tabular_vendor_eps_sweep.py --quick

  # Re-plot only (no GD re-run), optionally cropping x-axis:
  python run_tabular_vendor_eps_sweep.py --plot_only
  python run_tabular_vendor_eps_sweep.py --plot_only --z_max 20000
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth_utils import (
    SynthConfig, build_isotropic_operators, generate_dataset,
    train_population_gd, kappa_theory, save_logs, load_logs, ensure_dir,
)


# ============================================================
#  Configuration builder
# ============================================================

def make_eps_configs(alpha, epsilons, steps, N, noise_R, out_root):
    """
    For a fixed target alpha, build one SynthConfig per epsilon value.

    Strategy (same as alpha sweep): mu_A = mu_B = 1.0, mu = 0.0,
    gamma_min = 2 * alpha  =>  alpha = gamma_min / 2.
    """
    configs = []
    for eps in epsilons:
        cfg = SynthConfig(
            mu_A=1.0,
            mu_B=1.0,
            mu=0.0,
            gamma_min=2.0 * alpha,
            gamma_maj=1.0,
            epsilon=eps,
            noise_R=noise_R,
            N=N,
            steps=steps,
            lr=0.01,
            log_every=max(1, steps // 2000),
            print_every=steps // 10,
            out_dir=os.path.join(out_root, f"alpha_{alpha:.2f}",
                                 f"eps_{eps:.3f}"),
        )
        assert abs(cfg.alpha - alpha) < 1e-10, \
            f"Alpha mismatch: expected {alpha}, got {cfg.alpha}"
        configs.append(cfg)
    return configs


# ============================================================
#  Plotting
# ============================================================

# Distinct colours for each epsilon value
EPS_COLORS = {
    0.01:  "#d62728",   # red
    0.05:  "#ff7f0e",   # orange
    0.1:   "#2ca02c",   # green
    0.2:   "#1f77b4",   # blue
    0.5:   "#9467bd",   # purple
}

def _color(eps):
    return EPS_COLORS.get(eps, "black")


def make_eps_sweep_plots(results_by_eps, alpha, theory_by_eps, out_dir,
                         z_max=None):
    """
    For one fixed alpha, overlay curves for all epsilon values.

    Produces 4 plots:
      1. Raw minority error vs z_t  (log-log)
      2. Raw majority error vs z_t  (log-log)
      3. Rescaled minority error:  eps * err_min * z_t  vs z_t
         (should collapse for alpha < 1, fan out for alpha >= 1 — wait,
          actually: rescaled should collapse for BOTH regimes on majority,
          and collapse for alpha < 1 on minority, and collapse trivially
          for alpha >= 1 on minority since rate is eps-independent)
      4. Rescaled majority error:  (1-eps) * err_maj * z_t  vs z_t
    """
    ensure_dir(out_dir)

    # Prepare data for each epsilon
    plot_data = {}
    for eps, res in sorted(results_by_eps.items()):
        logs = res["logs"]
        cfg = res["cfg"]
        t = np.array(logs["t"], dtype=np.float64)
        err_maj = np.array(logs["err_maj"])
        err_min = np.array(logs["err_min"])
        z_t = cfg.lr * t

        ERR_FLOOR = 1e-9
        mask = (err_maj > ERR_FLOOR) & (err_min > ERR_FLOOR) & (z_t > 1.0)
        if z_max is not None:
            mask &= (z_t <= z_max)

        plot_data[eps] = {
            "z_t": z_t[mask], "err_maj": err_maj[mask],
            "err_min": err_min[mask], "cfg": cfg,
        }

    # --- Plot 1: Raw minority error (log-log) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps, d in sorted(plot_data.items()):
        ax.plot(d["z_t"], d["err_min"], color=_color(eps), lw=1.5,
                label=f"$\\varepsilon = {eps}$")
    # Reference slope
    z_ref = np.logspace(2, 4.5, 100)
    if alpha < 1:
        ax.plot(z_ref, 0.5 / z_ref, "k--", lw=1, alpha=0.4,
                label="$\\propto 1/z_t$")
    else:
        ax.plot(z_ref, 50 * z_ref**(-alpha), "k--", lw=1, alpha=0.4,
                label=f"$\\propto z_t^{{-{alpha:.1f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$z_t = h \cdot t$")
    ax.set_ylabel(r"$\mathrm{err}_{\min}$")
    if alpha < 1:
        ax.set_title(f"Minority error ($\\alpha = {alpha:.2f} < 1$): "
                     f"curves fan out $\\propto 1/\\varepsilon$")
    else:
        ax.set_title(f"Minority error ($\\alpha = {alpha:.2f} \\geq 1$): "
                     f"curves collapse ($\\varepsilon$-independent)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "min_error_raw.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "min_error_raw.pdf"))
    plt.close(fig)

    # --- Plot 2: Raw majority error (log-log) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps, d in sorted(plot_data.items()):
        ax.plot(d["z_t"], d["err_maj"], color=_color(eps), lw=1.5,
                label=f"$\\varepsilon = {eps}$")
    ax.plot(z_ref, 0.5 / z_ref, "k--", lw=1, alpha=0.4,
            label="$\\propto 1/z_t$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$z_t = h \cdot t$")
    ax.set_ylabel(r"$\mathrm{err}_{\mathrm{maj}}$")
    ax.set_title(f"Majority error ($\\alpha = {alpha:.2f}$): "
                 f"curves fan out $\\propto 1/(1-\\varepsilon)$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "maj_error_raw.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "maj_error_raw.pdf"))
    plt.close(fig)

    # --- Plot 3: Rescaled minority error ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps, d in sorted(plot_data.items()):
        rescaled = eps * d["err_min"] * d["z_t"]
        ax.plot(d["z_t"], rescaled, color=_color(eps), lw=1.5,
                label=f"$\\varepsilon = {eps}$")
    # Theory kappa_min lines (only for alpha < 1)
    if alpha < 1:
        # kappa_min is the same for all eps (it depends on geometry, not eps)
        any_theory = list(theory_by_eps.values())[0]
        if any_theory.get("kappa_min") is not None:
            ax.axhline(any_theory["kappa_min"], color="black", ls="--",
                       lw=1.5, alpha=0.6,
                       label=f"$\\kappa_{{\\min}} = "
                             f"{any_theory['kappa_min']:.4f}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$z_t = h \cdot t$")
    ax.set_ylabel(
        r"$\varepsilon \cdot \mathrm{err}_{\min} \cdot z_t$")
    if alpha < 1:
        ax.set_title(
            f"Rescaled minority ($\\alpha = {alpha:.2f} < 1$): "
            f"all $\\varepsilon$ collapse onto $\\kappa_{{\\min}}$")
        ax.set_ylim([0, 0.6])
    else:
        ax.set_title(
            f"Rescaled minority ($\\alpha = {alpha:.2f} \\geq 1$): "
            f"all $\\to 0$ ($\\varepsilon$-independent rate)")
        ax.set_ylim([0, 0.075])
    
    # Fixed y-axis for visual consistency
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "min_rescaled.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "min_rescaled.pdf"))
    plt.close(fig)

    # --- Plot 4: Rescaled majority error ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps, d in sorted(plot_data.items()):
        rescaled = (1 - eps) * d["err_maj"] * d["z_t"]
        ax.plot(d["z_t"], rescaled, color=_color(eps), lw=1.5,
                label=f"$\\varepsilon = {eps}$")
    # Theory kappa_maj (same for all eps)
    any_theory = list(theory_by_eps.values())[0]
    if any_theory.get("kappa_maj") is not None:
        ax.axhline(any_theory["kappa_maj"], color="black", ls="--",
                   lw=1.5, alpha=0.6,
                   label=f"$\\kappa_{{\\mathrm{{maj}}}} = "
                         f"{any_theory['kappa_maj']:.4f}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$z_t = h \cdot t$")
    ax.set_ylabel(
        r"$(1-\varepsilon) \cdot \mathrm{err}_{\mathrm{maj}} \cdot z_t$")
    ax.set_title(
        f"Rescaled majority ($\\alpha = {alpha:.2f}$): "
        f"all $\\varepsilon$ collapse onto $\\kappa_{{\\mathrm{{maj}}}}$")
    ax.set_ylim([0, 1.4])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "maj_rescaled.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "maj_rescaled.pdf"))
    plt.close(fig)


# ============================================================
#  Single-run driver
# ============================================================

def run_single(cfg):
    """Run GD for one configuration and return logs + theory."""
    print(f"\n{'='*60}")
    print(f"  alpha = {cfg.alpha:.3f}, epsilon = {cfg.epsilon}")
    print(f"  gamma_min={cfg.gamma_min:.2f}, "
          f"mu_A={cfg.mu_A}, mu_B={cfg.mu_B}, mu={cfg.mu}")
    print(f"  N = {cfg.N}, steps = {cfg.steps}")
    print(f"{'='*60}")

    A, B, v = build_isotropic_operators(cfg)
    X, groups, meta = generate_dataset(cfg, A, B, v)

    print(f"  Dataset: N={meta['N']}, n_maj={meta['n_maj']}, "
          f"n_min={meta['n_min']}")
    print(f"  Running GD ({cfg.steps} steps, lr={cfg.lr})...")

    logs, w_final = train_population_gd(cfg, X, groups)
    save_logs(logs, meta, cfg)

    theory = kappa_theory(cfg)
    print(f"  Theory: alpha={theory['alpha']:.3f}, "
          f"kappa_maj={theory['kappa_maj']:.4f}, "
          f"kappa_min={theory.get('kappa_min', 'N/A')}")
    print(f"  Final: err_maj={logs['err_maj'][-1]:.2e}, "
          f"err_min={logs['err_min'][-1]:.2e}")

    return logs, theory


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Epsilon sweep: Tabular + Vendor Scores")
    parser.add_argument("--out_dir",
                        default="./tabular_vendor_eps_sweep_results",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=10_000_000,
                        help="GD steps per configuration")
    parser.add_argument("--N", type=int, default=1_000_000,
                        help="Dataset size")
    parser.add_argument("--noise_R", type=float, default=0.1,
                        help="Vendor noise bound ||xi|| <= noise_R")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.7, 1.5],
                        help="Alpha values (one < 1, one > 1)")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.2, 0.5],
                        help="Epsilon values to sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check (fewer steps/samples)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Re-generate plots from saved logs")
    parser.add_argument("--z_max", type=float, default=None,
                        help="Crop x-axis at this z_t value")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.steps = 100_000
        args.N = 50_000
        args.epsilons = [0.05, 0.1, 0.5]
        args.alphas = [0.7, 1.5]

    print("=" * 60)
    print("  Epsilon Sweep: Tabular Data + Third-Party Vendor Features")
    print("=" * 60)
    print(f"  Output:   {args.out_dir}")
    print(f"  Alphas:   {args.alphas}")
    print(f"  Epsilons: {args.epsilons}")
    if args.plot_only:
        print(f"  Mode: PLOT ONLY")
    else:
        print(f"  Steps: {args.steps}, N: {args.N}")
    if args.z_max is not None:
        print(f"  z_max crop: {args.z_max:.0f}")

    for alpha in args.alphas:
        alpha_dir = os.path.join(args.out_dir, f"alpha_{alpha:.2f}")

        if args.plot_only:
            # Reload from saved logs
            results_by_eps = {}
            theory_by_eps = {}
            for eps in args.epsilons:
                run_dir = os.path.join(alpha_dir, f"eps_{eps:.3f}")
                logs_path = os.path.join(run_dir, "logs.json")
                if not os.path.exists(logs_path):
                    print(f"  [SKIP] No logs for alpha={alpha:.2f}, "
                          f"eps={eps}")
                    continue
                payload = load_logs(run_dir)
                logs = payload["logs"]
                cfg_dict = payload["config"]
                cfg = SynthConfig(
                    mu_A=cfg_dict["mu_A"], mu_B=cfg_dict["mu_B"],
                    mu=cfg_dict["mu"],
                    gamma_min=cfg_dict["gamma_min"],
                    gamma_maj=cfg_dict["gamma_maj"],
                    epsilon=cfg_dict["epsilon"],
                    noise_R=cfg_dict.get("noise_R", 0.1),
                    N=cfg_dict["N"], steps=cfg_dict["steps"],
                    lr=cfg_dict["lr"],
                    log_every=cfg_dict.get("log_every", 1),
                    print_every=cfg_dict.get("print_every", 1),
                    out_dir=run_dir,
                )
                theory = kappa_theory(cfg)
                results_by_eps[eps] = {"cfg": cfg, "logs": logs}
                theory_by_eps[eps] = theory
                print(f"  Loaded alpha={alpha:.2f}, eps={eps}")
        else:
            # Run GD for each epsilon
            configs = make_eps_configs(
                alpha=alpha, epsilons=args.epsilons,
                steps=args.steps, N=args.N, noise_R=args.noise_R,
                out_root=args.out_dir,
            )
            for cfg in configs:
                cfg.seed = args.seed

            results_by_eps = {}
            theory_by_eps = {}
            for cfg in configs:
                logs, theory = run_single(cfg)
                results_by_eps[cfg.epsilon] = {"cfg": cfg, "logs": logs}
                theory_by_eps[cfg.epsilon] = theory

        # Generate plots for this alpha
        if results_by_eps:
            make_eps_sweep_plots(results_by_eps, alpha, theory_by_eps,
                                alpha_dir, z_max=args.z_max)
            print(f"  Plots saved to {alpha_dir}/")

    # Save summary
    if not args.plot_only:
        summary = {}
        for alpha in args.alphas:
            alpha_dir = os.path.join(args.out_dir, f"alpha_{alpha:.2f}")
            for eps_dir in sorted(os.listdir(alpha_dir)):
                if not eps_dir.startswith("eps_"):
                    continue
                logs_path = os.path.join(alpha_dir, eps_dir, "logs.json")
                if os.path.exists(logs_path):
                    payload = load_logs(os.path.join(alpha_dir, eps_dir))
                    logs = payload["logs"]
                    cfg_dict = payload["config"]
                    key = f"alpha_{alpha:.2f}_{eps_dir}"
                    summary[key] = {
                        "alpha": alpha,
                        "epsilon": cfg_dict["epsilon"],
                        "final_err_maj": logs["err_maj"][-1],
                        "final_err_min": logs["err_min"][-1],
                    }
        summary_path = os.path.join(args.out_dir, "summary.json")
        ensure_dir(args.out_dir)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
