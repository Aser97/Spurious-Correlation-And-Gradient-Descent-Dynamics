"""
build_colored_mnist.py
----------------------
Build a colored-MNIST dataset with feature-mediated spurious correlation.

Data model:
  r      = raw MNIST digit image (28x28 grayscale)
  y      = 0 if digit in {0,..,4},  1 if digit in {5,..,9}
  digit_class = 0, 1, ..., 9

  Background intensity is a deterministic function of digit class:
      intensity_A = floor + (1-floor) * digit_class / 9   (increases 0->9)
      intensity_B = floor + (1-floor) * (1 - digit_class / 9)  (decreases 0->9)

  The mapping from r to intensity is nonlinear (digit class is a nonlinear
  function of raw pixels), but intensity is perfectly monotonic in digit class.

  For each image, independently of y:
      with prob 1-epsilon:  group = 0 (majority),  background = MAROON * intensity_A
      with prob   epsilon:  group = 1 (minority),  background = TEAL   * intensity_B

  After computing the background, the digit foreground is masked and overlaid
  so that the final image has a colored background and a grayscale digit.

Outputs (in --out_dir):
  colored_mnist.npz   — full dataset (images, labels, groups, etc.)
  showcase_colored_mnist.png — 2-row image for the manuscript

Usage (from Implicit-Bias-Spurious/):
    python build_colored_mnist.py [--epsilon 0.1] [--out_dir ./data_colored_mnist]
"""

import os
import argparse
import numpy as np
from PIL import Image

# torchvision only used for downloading MNIST
from torchvision import datasets


# ============================================================
#  Color palette
# ============================================================
# Maroon and teal at full intensity (will be scaled by [floor, 1])
MAROON = np.array([230, 0, 0], dtype=np.float32)
TEAL   = np.array([0, 230, 230], dtype=np.float32)


# ============================================================
#  Core: compute intensities from digit class and build images
# ============================================================

def compute_intensities(digit_classes, floor=0.3):
    """
    Compute background intensities from digit class (0-9).

    intensity_A = floor + (1-floor) * digit_class/9   — increases with digit
    intensity_B = floor + (1-floor) * (1 - digit_class/9)  — decreases with digit

    Parameters
    ----------
    digit_classes : ndarray (N,), int, values 0-9
    floor : float, minimum intensity to avoid pure black backgrounds

    Returns
    -------
    intensity_A, intensity_B : ndarray (N,), values in [floor, 1]
    """
    normed = (digit_classes.astype(np.float32)+1) / 10.0          # [0, 1]
    intensity_A = floor + (1 - floor) * normed               # [floor, 1]
    intensity_B = floor + (1 - floor) * (1 - normed)         # [floor, 1]
    return intensity_A, intensity_B


def build_colored_images(all_images, fg_mask, groups,
                         intensity_A, intensity_B):
    """
    Construct RGB images: grayscale digit foreground on colored background.

    Parameters
    ----------
    all_images : ndarray (N, 28, 28), uint8, original MNIST
    fg_mask    : ndarray (N, 28, 28), bool, True = foreground pixel
    groups     : ndarray (N,), int, 0 = majority (maroon), 1 = minority (teal)
    intensity_A, intensity_B : ndarray (N,), floats in [floor, 1]

    Returns
    -------
    colored : ndarray (N, 28, 28, 3), uint8
    """
    N = len(all_images)
    img_float = all_images.astype(np.float32)    # (N, 28, 28), 0-255
    colored = np.zeros((N, 28, 28, 3), dtype=np.float32)

    maj = (groups == 0)
    mino = (groups == 1)

    # Majority: maroon background
    if maj.any():
        for c in range(3):
            bg = MAROON[c] * intensity_A[maj, None, None]  # (n_maj, 1, 1)
            fg = img_float[maj]                             # (n_maj, 28, 28)
            colored[maj, :, :, c] = np.where(fg_mask[maj], fg, bg)

    # Minority: teal background
    if mino.any():
        for c in range(3):
            bg = TEAL[c] * intensity_B[mino, None, None]
            fg = img_float[mino]
            colored[mino, :, :, c] = np.where(fg_mask[mino], fg, bg)

    return np.clip(colored, 0, 255).astype(np.uint8)


# ============================================================
#  Showcase image for the manuscript
# ============================================================

def build_showcase(all_images, all_labels, fg_mask,
                   intensity_A, intensity_B, out_dir, pad=2, scale=5):
    """
    Build a 2-row showcase image:
      Top row:    digits 0-9 with MAROON background (shade increasing)
      Bottom row: digits 0-9 with TEAL   background (shade decreasing)

    Picks one representative image per digit (median intensity).
    """
    cell_h, cell_w = 28 + 2 * pad, 28 + 2 * pad
    canvas = np.zeros((2 * cell_h, 10 * cell_w, 3), dtype=np.uint8)

    for row, (color_base, intensities) in enumerate([
        (MAROON, intensity_A),
        (TEAL,   intensity_B),
    ]):
        for col, digit in enumerate(range(10)):
            indices = np.where(all_labels == digit)[0]
            digit_int = intensities[indices]
            # Pick the image closest to the median intensity for this digit
            med_idx = indices[np.argsort(digit_int)[len(digit_int) // 2]]

            img_gray = all_images[med_idx].astype(np.float32)  # (28, 28)
            mask = fg_mask[med_idx]                             # (28, 28)
            bg_color = color_base * intensities[med_idx]        # (3,)

            rgb = np.zeros((28, 28, 3), dtype=np.float32)
            for c in range(3):
                rgb[:, :, c] = np.where(mask, img_gray, bg_color[c])

            cell = np.clip(rgb, 0, 255).astype(np.uint8)
            y0 = row * cell_h + pad
            x0 = col * cell_w + pad
            canvas[y0:y0+28, x0:x0+28] = cell

    # Scale up for visibility
    pil = Image.fromarray(canvas)
    pil = pil.resize((canvas.shape[1] * scale, canvas.shape[0] * scale),
                     Image.NEAREST)
    path = os.path.join(out_dir, "showcase_colored_mnist.png")
    pil.save(path)
    print(f"Showcase image saved to {path}")
    return path


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build colored-MNIST dataset with feature-mediated spurious correlation.")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Minority group fraction")
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Std of Gaussian noise added to intensity (xi)")
    parser.add_argument("--out_dir", type=str, default="./data_colored_mnist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--intensity_floor", type=float, default=0.15,
                        help="Minimum background intensity to keep colors visible")
    parser.add_argument("--fg_threshold", type=int, default=20,
                        help="Pixel threshold (0-255) for foreground mask")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    #  Load MNIST
    # ------------------------------------------------------------------
    print("[1/4] Loading MNIST...")
    train_set = datasets.MNIST(root=os.path.join(args.out_dir, "mnist_raw"),
                               train=True, download=True)
    test_set  = datasets.MNIST(root=os.path.join(args.out_dir, "mnist_raw"),
                               train=False, download=True)

    all_images = np.concatenate([train_set.data.numpy(),
                                 test_set.data.numpy()], axis=0)   # (70000, 28, 28)
    all_labels = np.concatenate([train_set.targets.numpy(),
                                 test_set.targets.numpy()], axis=0) # (70000,)
    N = len(all_images)
    print(f"  Total images: {N}")

    # ------------------------------------------------------------------
    #  Compute intensities from digit class
    # ------------------------------------------------------------------
    print("[2/4] Computing intensities from digit class...")
    intensity_A, intensity_B = compute_intensities(
        all_labels, floor=args.intensity_floor)
    print(f"  Intensity A range: [{intensity_A.min():.3f}, {intensity_A.max():.3f}]")
    print(f"  Intensity B range: [{intensity_B.min():.3f}, {intensity_B.max():.3f}]")

    # Optional noise on intensity
    if args.noise_std > 0:
        xi = np.random.randn(N).astype(np.float32) * args.noise_std
        intensity_A = np.clip(intensity_A + xi, args.intensity_floor, 1.0)
        intensity_B = np.clip(intensity_B - xi, args.intensity_floor, 1.0)
        print(f"  Added noise xi ~ N(0, {args.noise_std}^2) to intensities")

    # ------------------------------------------------------------------
    #  Assign groups and labels
    # ------------------------------------------------------------------
    print("[3/4] Assigning groups and labels...")
    groups = (np.random.rand(N) < args.epsilon).astype(np.int64)
    y = (all_labels >= 5).astype(np.int64)
    print(f"  Majority (maroon): {(groups==0).sum()}")
    print(f"  Minority (teal):   {(groups==1).sum()}")
    print(f"  y=0 (digits 0-4):  {(y==0).sum()}")
    print(f"  y=1 (digits 5-9):  {(y==1).sum()}")

    # ------------------------------------------------------------------
    #  Build colored images
    # ------------------------------------------------------------------
    print("[4/4] Building colored images...")
    fg_mask = all_images > args.fg_threshold   # (N, 28, 28)
    colored = build_colored_images(all_images, fg_mask, groups,
                                   intensity_A, intensity_B)
    print(f"  Image tensor shape: {colored.shape}")

    # ------------------------------------------------------------------
    #  Save
    # ------------------------------------------------------------------
    print("Saving...")

    # Save the actual intensity used per sample (depends on group)
    intensity_used = np.where(groups == 0, intensity_A, intensity_B)

    save_path = os.path.join(args.out_dir, "colored_mnist.npz")
    np.savez_compressed(
        save_path,
        images=colored,              # (N, 28, 28, 3) uint8
        y=y,                         # (N,) binary label
        groups=groups,               # (N,) 0=majority/maroon, 1=minority/teal
        digit_classes=all_labels,    # (N,) 0-9
        intensity_used=intensity_used,  # (N,) actual bg intensity for this sample
        intensity_A=intensity_A,     # (N,) intensity under operator A
        intensity_B=intensity_B,     # (N,) intensity under operator B
        epsilon=np.float32(args.epsilon),
        noise_std=np.float32(args.noise_std),
        seed=np.int32(args.seed),
    )
    print(f"  Dataset saved to {save_path}")

    # Showcase image (uses all images, not just one group)
    build_showcase(all_images, all_labels, fg_mask,
                   intensity_A, intensity_B, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
