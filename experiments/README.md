# Experiments: Implicit Bias of GD under Spurious Correlations

> **Note on this copy.** Paths below refer to the layout in this repository:
> cropped α-sweep figures are committed under `figures/` (PNG only), and the raw
> `logs.json` files are committed under `tabular_vendor_results/` and
> `tabular_vendor_eps_sweep_results/` so every plot can be regenerated without
> re-running gradient descent. The Colored-MNIST dataset (`data_colored_mnist/`,
> ~188 MB) is **not** committed — rebuild it with `build_colored_mnist.py`.

This directory contains two experiment suites:

1. **Tabular + Vendor Score** — synthetic experiments verifying Theorems 1 & 2 (isotropic regime).
2. **Colored MNIST + SAE Feature Analysis** — empirical evidence that nonlinear feature-mediated correlations linearise in learned representations (Appendix B).

## Setup

```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn
# For GPU support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

GPU is optional but recommended for large N. The code auto-detects CUDA.

---

## Part 1: Tabular + Vendor Score (Theorems 1 & 2)

### Scripts

| Script | Purpose |
|--------|---------|
| `synth_utils.py` | Core library: `SynthConfig`, data generation, GD training, theory predictions |
| `run_tabular_vendor.py` | **α-sweep**: fix ε=0.1, sweep α ∈ {0.5, 0.7, 1.0, 1.3, 1.5, 2.0} |
| `run_tabular_vendor_eps_sweep.py` | **ε-sweep**: fix α ∈ {0.7, 1.5}, sweep ε ∈ {0.01, 0.05, 0.1, 0.2, 0.5} |
| `replot_tabular_vendor.py` | Re-generate α-sweep plots with x-axis cropping (no GD re-run) |

## Quick sanity check (< 5 min, CPU is fine)

```bash
python run_tabular_vendor.py --quick
python run_tabular_vendor_eps_sweep.py --quick
```

## Full runs (for the paper)

### α-sweep (verifies asymptotic rates + phase transition)

```bash
# Default: 10M steps, N=5M, h=0.01, 6 alpha values (~several hours on GPU)
python run_tabular_vendor.py

# Re-plot with x-axis crop at z_t = 20000 (instant, no GD):
python replot_tabular_vendor.py --T_max 20000
```

Produces per-α plots (rescaled errors, compensated, error ratio) in
`figures/`, plus an α-sweep overlay and phase transition
summary.

### ε-sweep (verifies ε-dependence and ε-independence)

```bash
# Default: 10M steps, N=1M, 2 alphas × 5 epsilons
python run_tabular_vendor_eps_sweep.py

# Resume if interrupted (only run missing configs):
python run_tabular_vendor_eps_sweep.py --alphas 1.5 --epsilons 0.1 0.2 0.5

# Re-plot from saved logs:
python run_tabular_vendor_eps_sweep.py --plot_only
```

Produces minority/majority raw and rescaled error overlays in
`tabular_vendor_eps_sweep_results/alpha_*/`.

## Data model

All experiments use the **Tabular + Vendor Score** instantiation of the
isotropic regime (Definition 2 in the paper):

- Raw causal features r ∈ ℝ^{d_r} with label-absorbed margins
- Vendor risk scores s = A·r + ξ (majority) or s = B·r + ξ (minority)
- A, B share the same singular-vector basis but differ in singular values
- Full feature vector x = [r; s] ∈ ℝ^{d_r + d_s}

Parameters: μ_A = μ_B = 1, μ = 0, γ_min = 2·α. This isolates the margin
advantage as the sole driver of the phase transition.

## File structure

```
.
├── synth_utils.py                          # Core library
├── run_tabular_vendor.py                   # α-sweep experiment
├── run_tabular_vendor_eps_sweep.py         # ε-sweep experiment
├── replot_tabular_vendor.py                # Replotting with crop/overlay
├── tabular_vendor_results/                 # α-sweep raw logs
│   └── alpha_*/logs.json
├── figures/            # α-sweep cropped figures
│   ├── alpha_*/rescaled_errors.{png,pdf}
│   ├── alpha_sweep_overlay.{png,pdf}
│   └── phase_transition_summary.{png,pdf}
└── tabular_vendor_eps_sweep_results/       # ε-sweep results
    ├── alpha_0.70/eps_*/logs.json
    ├── alpha_0.70/{min,maj}_{error_raw,rescaled}.{png,pdf}
    ├── alpha_1.50/eps_*/logs.json
    └── alpha_1.50/{min,maj}_{error_raw,rescaled}.{png,pdf}
```

## Expected results

**α-sweep (ε = 0.1):** For α < 1, both rescaled errors plateau at κ_g. For
α ≥ 1, majority plateaus while minority → 0. The α-sweep overlay shows all
curves compensated by z_t^{max(1,α)} approximately plateau. The phase
transition summary shows the measured decay exponent tracking the theoretical
prediction with a kink at α = 1.

**ε-sweep (α = 0.7):** Minority raw errors fan out ∝ 1/ε; rescaled errors
collapse onto κ_min regardless of ε.

**ε-sweep (α = 1.5):** Minority raw errors collapse (ε-independent rate
z_t^{-1.5}); majority errors fan out ∝ 1/(1−ε).

---

## Part 2: Colored MNIST + SAE Feature Analysis (Appendix B)

This experiment tests whether a nonlinear feature-mediated spurious correlation
becomes linear in a learned representation. A colored-MNIST dataset is
constructed where background intensity depends on digit class (a nonlinear
function of raw pixels) with group-specific direction (majority: increasing,
minority: decreasing). An MLP is trained, representations are extracted at
early checkpoints, and Sparse Autoencoders (SAEs) decompose them into
causal (r) and spurious (s) features. A Ridge regression φ_r → φ_s is
fit per group, with a permutation test as null-distribution control.

### Scripts

| Script | Purpose |
|--------|---------|
| `build_colored_mnist.py` | Build the colored-MNIST dataset (N=70k, ε=0.1) |
| `train_mlp_colored_mnist.py` | Train MLP with checkpoints at epochs 1–5 |
| `train_sae_analyze.py` | Train SAE, classify features, measure linearity, permutation test |

### Quick run

```bash
# 1. Build dataset
python build_colored_mnist.py --epsilon 0.1 --out_dir ./data_colored_mnist

# 2. Train MLP with checkpoints
python train_mlp_colored_mnist.py --data_dir ./data_colored_mnist --checkpoint_epochs 1,2,3,4,5

# 3. Run SAE analysis on all checkpoints (with 1000-permutation null test)
python train_sae_analyze.py --data_dir ./data_colored_mnist --run_all_epochs --alpha 0.03 --n_perm 1000
```

### Data model

Each MNIST image receives a colored background whose intensity is a
deterministic function of digit class:

- **Majority (maroon, 90%):** intensity = floor + (1−floor) · (digit+1)/10 (increases with digit)
- **Minority (teal, 10%):** intensity = floor + (1−floor) · (1 − (digit+1)/10) (decreases with digit)

The label is y = 1[digit ≥ 5]. The digit foreground is grayscale; only the
background carries the spurious signal. The spurious feature depends on the
*value* of the causal feature (digit class), not merely on y — making this a
feature-mediated spurious correlation with a nonlinear input-space map.

### SAE feature classification

Each SAE feature c_k is classified using group-conditional correlations with
digit class d ∈ {0,…,9}:

- ρ₀(k) = corr(c_k, d | g=0), ρ₁(k) = corr(c_k, d | g=1)
- Same sign → r-feature (tracks digit shape, invariant to color)
- Opposite sign → s-feature (tracks background color, flips with group)
- Threshold |ρ| ≥ 0.2 for classification

### File structure

```
data_colored_mnist/
├── colored_mnist.npz                    # Full dataset (70k images)
├── showcase_colored_mnist.png           # 2-row sample image (manuscript figure)
├── sae_all_epochs_summary.json          # Combined results across all epochs
├── mnist_raw/                           # Downloaded MNIST
└── epoch_{1..5}/
    ├── mlp_model.pt                     # MLP checkpoint
    ├── representations.npz              # 128-dim penultimate activations φ(x)
    ├── sae_model.pt                     # Trained SAE
    ├── sae_summary.json                 # Per-epoch results (R², null test, etc.)
    ├── sae_feature_correlations.png     # (ρ₀, ρ₁) scatter plot
    └── phi_r_vs_phi_s.png              # First PC of φ_r vs φ_s, by group
```

### Expected results

- **Minority R²:** saturates near 1.0 from epoch 2 onward (near-perfect linear φ_r → φ_s map).
- **Majority R²:** lower and variable (0.38–0.79), but 3 orders of magnitude above the null baseline (~0.0001–0.0005).
- **Permutation test:** all p-values < 0.001 (1000 permutations), confirming the linear relationship is genuine.
- **φ_r vs φ_s scatter:** minority forms a tight linear cloud with clear slope; majority is flat near φ_s ≈ 0. Slope sign alternates across epochs (SVD sign ambiguity — cosmetic, does not affect R²).

---

## Compute resources

**Tabular experiments:** all run on a single NVIDIA RTX A6000 GPU (49 GB).
The α-sweep (N=1M, 10M steps, 6 configs) takes approximately 2–4 hours.
The ε-sweep (N=1M, 10M steps, 10 configs) takes approximately 4–6 hours.

**Colored MNIST:** runs on CPU in ~10 minutes (dataset build + MLP training +
SAE analysis across 5 epochs). The permutation test (1000 permutations × 5
epochs × 3 groups) adds ~5 minutes.
