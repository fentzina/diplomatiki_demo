#!/usr/bin/env python3
"""
validate_tumor_pancreas_rois.py

QC check on the ROIs produced by allilouia.py: for every PDAC case
(label 1 present), quantify the spatial relationship between the tumor
mask (label 1) and the pancreas mask (label 4).

For each case, reports:
  - n_label1, n_label4          : raw voxel counts
  - n_overlap                   : voxels where label 1 and label 4 coincide exactly
  - has_overlap                 : bool, n_overlap > 0
  - n_tumor_completely_outside  : label-1 voxels with NO label-4 voxel at that
                                   exact location (i.e. n_label1 - n_overlap)
  - n_tumor_isolated            : label-1 voxels with NO label-4 voxel anywhere
                                   in their immediate 3x3x3 neighbourhood
                                   (stricter than "completely outside" -- a
                                   tumor voxel can be non-overlapping but still
                                   directly adjacent to pancreas, and would NOT
                                   count here)
  - iou_local                   : IoU between label 1 and ONLY the connected
                                   component(s) of label 4 that actually touch
                                   label 1 (i.e. the specific pancreas blob the
                                   tumor sits in/against, not unrelated
                                   disconnected pancreas fragments elsewhere in
                                   the volume)

Across the full dataset it also reports how many of the N PDAC cases have
any label1/label4 overlap at all.

Usage:
    python validate_tumor_pancreas_rois.py \
        --label_dir /path/to/panorama_labels/automatic_labels \
        --out_dir ./roi_validation
"""

import os
import csv
import glob
import argparse

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TUMOR_LABEL = 1
PANCREAS_LABEL = 4

# 26-connectivity: full 3x3x3 neighbourhood (includes face/edge/corner neighbours)
NEIGHBOURHOOD_STRUCT = np.ones((3, 3, 3), dtype=bool)


def load_mask(mask_path):
    img = nib.load(mask_path)
    data = np.asarray(img.get_fdata())
    return np.rint(data).astype(np.int32)


def find_label_files(path):
    if os.path.isfile(path):
        return [path]
    patterns = [os.path.join(path, "**", "*.nii"), os.path.join(path, "**", "*.nii.gz")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(files)


def case_stem_of(path):
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return base.split(".")[0]


def analyze_case(mask):
    tumor = (mask == TUMOR_LABEL)
    pancreas = (mask == PANCREAS_LABEL)

    n_tumor = int(tumor.sum())
    if n_tumor == 0:
        return None  # not a PDAC case here

    n_pancreas = int(pancreas.sum())
    overlap = tumor & pancreas
    n_overlap = int(overlap.sum())

    # --- "completely outside": exact voxel-level, no overlap at that location ---
    n_completely_outside = n_tumor - n_overlap

    # --- "isolated": no label-4 voxel anywhere in the immediate 3x3x3 neighbourhood ---
    if n_pancreas > 0:
        pancreas_dilated = ndi.binary_dilation(pancreas, structure=NEIGHBOURHOOD_STRUCT)
        isolated = tumor & ~pancreas_dilated
        n_isolated = int(isolated.sum())
    else:
        n_isolated = n_tumor  # no pancreas at all -> every tumor voxel is isolated

    # --- local IoU: only against pancreas connected component(s) touching tumor ---
    if n_pancreas > 0:
        pancreas_cc, n_cc = ndi.label(pancreas, structure=NEIGHBOURHOOD_STRUCT)
        touching_labels = np.unique(pancreas_cc[tumor & (pancreas_cc > 0)])
        touching_labels = touching_labels[touching_labels > 0]
        local_pancreas = np.isin(pancreas_cc, touching_labels) if len(touching_labels) else np.zeros_like(pancreas)

        inter = int((tumor & local_pancreas).sum())
        union = int((tumor | local_pancreas).sum())
        iou_local = (inter / union) if union > 0 else float("nan")
        n_pancreas_components = int(n_cc)
        n_components_touching_tumor = int(len(touching_labels))
    else:
        iou_local = 0.0
        n_pancreas_components = 0
        n_components_touching_tumor = 0

    return {
        "n_label1": n_tumor,
        "n_label4": n_pancreas,
        "n_overlap": n_overlap,
        "has_overlap": n_overlap > 0,
        "n_tumor_completely_outside": n_completely_outside,
        "pct_tumor_completely_outside": 100 * n_completely_outside / n_tumor,
        "n_tumor_isolated": n_isolated,
        "pct_tumor_isolated": 100 * n_isolated / n_tumor,
        "iou_local": iou_local,
        "n_pancreas_components": n_pancreas_components,
        "n_components_touching_tumor": n_components_touching_tumor,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate label1 (tumor) vs label4 (pancreas) ROIs across all PDAC cases."
    )
    parser.add_argument("--label_dir", required=True,
                         help="Path to a single label file, or a directory of label files.")
    parser.add_argument("--out_dir", default=".", help="Where to save the CSV and plots.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    label_files = find_label_files(args.label_dir)
    if not label_files:
        print(f"No .nii/.nii.gz files found at {args.label_dir}")
        return
    print(f"Found {len(label_files)} label file(s). Checking PDAC cases (label 1 present)...")

    rows = []
    n_no_tumor = 0
    for fp in label_files:
        case_id = case_stem_of(fp)
        try:
            mask = load_mask(fp)
        except Exception as e:
            print(f"FAILED to load {fp}: {e}")
            continue

        result = analyze_case(mask)
        if result is None:
            n_no_tumor += 1
            continue

        result["case_id"] = case_id
        rows.append(result)

    if not rows:
        print("No PDAC cases (label 1 present) found. Nothing to report.")
        return

    # --- CSV ---
    csv_path = os.path.join(args.out_dir, "roi_validation_summary.csv")
    fieldnames = ["case_id", "n_label1", "n_label4", "n_overlap", "has_overlap",
                  "n_tumor_completely_outside", "pct_tumor_completely_outside",
                  "n_tumor_isolated", "pct_tumor_isolated",
                  "iou_local", "n_pancreas_components", "n_components_touching_tumor"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved per-case CSV -> {csv_path}")

    # --- Summary across all PDAC cases ---
    n_cases = len(rows)
    n_overlap_cases = sum(1 for r in rows if r["has_overlap"])
    n_no_overlap_cases = n_cases - n_overlap_cases

    pct_outside = np.array([r["pct_tumor_completely_outside"] for r in rows])
    pct_isolated = np.array([r["pct_tumor_isolated"] for r in rows])
    iou_vals = np.array([r["iou_local"] for r in rows])
    multi_comp_cases = sum(1 for r in rows if r["n_components_touching_tumor"] > 1)

    print("\n" + "=" * 72)
    print(f"PDAC cases evaluated (label 1 present)         : {n_cases}")
    print(f"  (skipped, no label 1 at all)                 : {n_no_tumor}")
    print("-" * 72)
    print(f"Cases WITH label1/label4 overlap               : {n_overlap_cases} / {n_cases} "
          f"({100*n_overlap_cases/n_cases:.1f}%)")
    print(f"Cases with NO label1/label4 overlap             : {n_no_overlap_cases} / {n_cases} "
          f"({100*n_no_overlap_cases/n_cases:.1f}%)")
    print("-" * 72)
    print(f"Tumor voxels 'completely outside' pancreas (exact, no overlap):")
    print(f"  mean {pct_outside.mean():.2f}%  median {np.median(pct_outside):.2f}%  max {pct_outside.max():.2f}%")
    print(f"Tumor voxels 'isolated' (no pancreas in immediate 3x3x3 neighbourhood):")
    print(f"  mean {pct_isolated.mean():.2f}%  median {np.median(pct_isolated):.2f}%  max {pct_isolated.max():.2f}%")
    print("-" * 72)
    print(f"Local IoU (tumor vs. only the touching pancreas component(s)):")
    print(f"  mean {np.nanmean(iou_vals):.4f}  median {np.nanmedian(iou_vals):.4f}  "
          f"min {np.nanmin(iou_vals):.4f}  max {np.nanmax(iou_vals):.4f}")
    print(f"Cases where tumor touches >1 disconnected pancreas component : {multi_comp_cases} / {n_cases}")
    print("=" * 72)

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(pct_outside, bins=30, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("% tumor voxels with no exact-location pancreas overlap")
    axes[0, 0].set_xlabel("%"); axes[0, 0].set_ylabel("N cases")

    axes[0, 1].hist(pct_isolated, bins=30, color="indianred", edgecolor="black")
    axes[0, 1].set_title("% tumor voxels isolated (no pancreas in 3x3x3 neighbourhood)")
    axes[0, 1].set_xlabel("%"); axes[0, 1].set_ylabel("N cases")

    axes[1, 0].hist(iou_vals[~np.isnan(iou_vals)], bins=30, color="seagreen", edgecolor="black")
    axes[1, 0].set_title("Local IoU (tumor vs. touching pancreas component)")
    axes[1, 0].set_xlabel("IoU"); axes[1, 0].set_ylabel("N cases")

    axes[1, 1].bar(["Overlap", "No overlap"], [n_overlap_cases, n_no_overlap_cases],
                   color=["seagreen", "gray"])
    for i, v in enumerate([n_overlap_cases, n_no_overlap_cases]):
        axes[1, 1].text(i, v, str(v), ha="center", va="bottom")
    axes[1, 1].set_title(f"Cases with label1/label4 overlap (of {n_cases} PDAC cases)")

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "roi_validation_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
