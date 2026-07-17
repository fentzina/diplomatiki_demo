import os
import argparse
import itertools
import numpy as np

parser = argparse.ArgumentParser(
    description="Diagnose ROI/mask misalignment by testing flips and axis permutations."
)
parser.add_argument("--coliage_dir", required=True,
                     help="Directory containing *_tensor_128_27ch.npy files")
parser.add_argument("--mask_dir", required=True,
                     help="Directory containing {case_id}_mask.npy files")
parser.add_argument("--n_cases", type=int, default=10,
                     help="Number of cases to check (default: 10)")
args = parser.parse_args()

COLIAGE_DIR = args.coliage_dir
MASK_DIR    = args.mask_dir
FILENAME_SUFFIX = "_tensor_128_27ch.npy"
N_CHANNELS = 26

coliage_files = sorted([
    f for f in os.listdir(COLIAGE_DIR) if f.endswith(FILENAME_SUFFIX)
])[:args.n_cases]

if len(coliage_files) == 0:
    raise FileNotFoundError(f"No {FILENAME_SUFFIX} files found in {COLIAGE_DIR}")

print(f"Checking {len(coliage_files)} cases for spatial misalignment...\n")

# All 3-axis permutations, and both flip states per axis after permutation
axis_perms = list(itertools.permutations([0, 1, 2]))
flip_combos = list(itertools.product([False, True], repeat=3))

# Track best transform per case, and vote across cases for the most common fix
best_transform_votes = {}

for filename in coliage_files:
    case_id = filename.replace(FILENAME_SUFFIX, "")
    coliage_path = os.path.join(COLIAGE_DIR, filename)
    mask_path = os.path.join(MASK_DIR, f"{case_id}_mask.npy")

    if not os.path.exists(mask_path):
        print(f"[{case_id}] SKIPPED — mask not found at {mask_path}")
        continue

    tensor = np.load(coliage_path).astype(np.float32)
    texture_channels = tensor[..., 1:1 + N_CHANNELS]
    roi = np.any(np.isfinite(texture_channels) & (texture_channels != 0.0), axis=-1)

    mask = np.load(mask_path).astype(bool)

    if roi.shape != mask.shape:
        print(f"[{case_id}] SHAPE MISMATCH — roi{roi.shape} vs mask{mask.shape}, "
              f"cannot test flips/transposes directly.")
        continue

    baseline_iou = (np.logical_and(roi, mask).sum() /
                     max(np.logical_or(roi, mask).sum(), 1))

    best_iou = baseline_iou
    best_transform = ("none", "none")

    for perm in axis_perms:
        transposed = np.transpose(mask, perm)
        for flips in flip_combos:
            candidate = transposed
            for axis, do_flip in enumerate(flips):
                if do_flip:
                    candidate = np.flip(candidate, axis=axis)
            overlap = np.logical_and(roi, candidate).sum()
            union = np.logical_or(roi, candidate).sum()
            iou = overlap / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_transform = (perm, flips)

    print(f"[{case_id}] baseline IoU: {baseline_iou:.3f}  |  "
          f"best IoU: {best_iou:.3f}  |  "
          f"best transform: transpose{best_transform[0]}, flips{best_transform[1]}")

    if best_iou > baseline_iou + 0.05:  # meaningful improvement
        key = str(best_transform)
        best_transform_votes[key] = best_transform_votes.get(key, 0) + 1

print("\n" + "=" * 60)
if best_transform_votes:
    print("Most common improving transform(s) across cases:")
    for k, v in sorted(best_transform_votes.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}/{len(coliage_files)} cases")
    print("\nIf one transform dominates, apply it to masks when loading, e.g.:")
    print("  mask = np.transpose(mask, PERM)")
    print("  for axis in FLIP_AXES: mask = np.flip(mask, axis=axis)")
else:
    print("No flip/transpose improved alignment meaningfully.")
    print("This suggests the issue is NOT a simple axis flip/transpose —")
    print("check resampling, cropping origin, or whether tensor/mask pairs")
    print("actually correspond to the same case.")
