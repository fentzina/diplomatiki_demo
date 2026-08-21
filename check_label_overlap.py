#!/usr/bin/env python3
"""
check_label_overlap.py

Checks whether label 1 (tumor) and label 4 (pancreas) occupy the same
voxels in a segmentation mask, reports voxel counts / overlap statistics,
and saves a diagnostic plot showing where (if anywhere) they intersect.

Usage:
    python check_label_overlap.py --mask_path /path/to/label.nii.gz
    python check_label_overlap.py --mask_path /path/to/label.nii.gz --label_a 1 --label_b 4
    python check_label_overlap.py --mask_path /path/to/label.nii.gz --out_dir ./overlap_check

If the mask is 3D, the plot shows the axial slice with the most overlap
(or the most combined foreground if there is no overlap at all).
"""

import os
import argparse

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_mask(mask_path):
    img = nib.load(mask_path)
    data = np.asarray(img.get_fdata())
    return np.rint(data).astype(np.int32)  # labels should be integers


def analyze_overlap(mask, label_a, label_b):
    mask_a = (mask == label_a)
    mask_b = (mask == label_b)

    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    overlap_mask = mask_a & mask_b
    n_overlap = int(overlap_mask.sum())

    union = int((mask_a | mask_b).sum())
    dice = (2 * n_overlap / (n_a + n_b)) if (n_a + n_b) > 0 else float("nan")
    iou = (n_overlap / union) if union > 0 else float("nan")
    pct_of_a_in_b = (100 * n_overlap / n_a) if n_a > 0 else float("nan")
    pct_of_b_in_a = (100 * n_overlap / n_b) if n_b > 0 else float("nan")

    return {
        "n_label_a": n_a,
        "n_label_b": n_b,
        "n_overlap": n_overlap,
        "union": union,
        "dice": dice,
        "iou": iou,
        "pct_of_a_in_b": pct_of_a_in_b,
        "pct_of_b_in_a": pct_of_b_in_a,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "overlap_mask": overlap_mask,
    }


def print_report(stats, label_a, label_b):
    print("=" * 60)
    print(f"Label {label_a} voxel count : {stats['n_label_a']}")
    print(f"Label {label_b} voxel count : {stats['n_label_b']}")
    print(f"Overlapping voxels        : {stats['n_overlap']}")
    print(f"Union voxels               : {stats['union']}")
    print("-" * 60)
    if stats["n_overlap"] > 0:
        print(f"Labels {label_a} and {label_b} DO overlap.")
        print(f"  Dice coefficient      : {stats['dice']:.4f}")
        print(f"  IoU (Jaccard)         : {stats['iou']:.4f}")
        print(f"  % of label {label_a} inside label {label_b} : {stats['pct_of_a_in_b']:.2f}%")
        print(f"  % of label {label_b} inside label {label_a} : {stats['pct_of_b_in_a']:.2f}%")
    else:
        print(f"Labels {label_a} and {label_b} do NOT overlap (0 shared voxels).")
    print("=" * 60)


def best_slice_for_plot(stats):
    """Pick the axial (z) slice with the most overlap; fall back to the
    slice with the most combined foreground if there's no overlap at all."""
    overlap_per_slice = stats["overlap_mask"].sum(axis=(0, 1))
    if overlap_per_slice.max() > 0:
        return int(np.argmax(overlap_per_slice))
    combined_per_slice = (stats["mask_a"] | stats["mask_b"]).sum(axis=(0, 1))
    return int(np.argmax(combined_per_slice))


def make_plot(mask, stats, label_a, label_b, out_path):
    is_3d = mask.ndim == 3
    z = best_slice_for_plot(stats) if is_3d else None

    if is_3d:
        slice_a = stats["mask_a"][:, :, z]
        slice_b = stats["mask_b"][:, :, z]
        slice_overlap = stats["overlap_mask"][:, :, z]
    else:
        slice_a = stats["mask_a"]
        slice_b = stats["mask_b"]
        slice_overlap = stats["overlap_mask"]

    # RGB composite: label_a = red, label_b = blue, overlap = magenta/white
    h, w = slice_a.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = slice_a  # red channel <- label A
    rgb[..., 2] = slice_b  # blue channel <- label B
    # overlap ends up as red+blue = magenta automatically

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    axes[0].imshow(rgb)
    axes[0].set_title(
        f"Label {label_a} (red) vs Label {label_b} (blue)"
        + (f"\nslice z={z}" if is_3d else "")
    )
    axes[0].axis("off")

    labels = [f"Label {label_a}", f"Label {label_b}", "Overlap"]
    counts = [stats["n_label_a"], stats["n_label_b"], stats["n_overlap"]]
    colors = ["red", "blue", "magenta"]
    axes[1].bar(labels, counts, color=colors)
    for i, c in enumerate(counts):
        axes[1].text(i, c, str(c), ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Voxel count (whole volume)")
    axes[1].set_title("Voxel counts")

    fig.suptitle(
        f"Dice={stats['dice']:.3f}  IoU={stats['iou']:.3f}"
        if stats["n_overlap"] > 0 else "No overlap between labels",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check whether two labels in a segmentation mask overlap, "
                    "report voxel counts, and plot the result."
    )
    parser.add_argument("--mask_path", required=True, help="Path to the label/segmentation .nii or .nii.gz file")
    parser.add_argument("--label_a", type=int, default=1, help="First label to check (default: 1 = tumor)")
    parser.add_argument("--label_b", type=int, default=4, help="Second label to check (default: 4 = pancreas)")
    parser.add_argument("--out_dir", default=".", help="Directory to save the plot into (default: current dir)")
    args = parser.parse_args()

    mask = load_mask(args.mask_path)
    stats = analyze_overlap(mask, args.label_a, args.label_b)
    print_report(stats, args.label_a, args.label_b)

    os.makedirs(args.out_dir, exist_ok=True)
    case_stem = os.path.basename(args.mask_path).split(".")[0]
    out_path = os.path.join(args.out_dir, f"{case_stem}_label{args.label_a}_vs_label{args.label_b}_overlap.png")
    make_plot(mask, stats, args.label_a, args.label_b, out_path)


if __name__ == "__main__":
    main()
