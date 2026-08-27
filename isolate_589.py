import os
import shutil

# ── CONFIG ────────────────────────────────────────────────────────────────────
VALID_IDS_FILE = "/home/student1/ftzina_thesis/output_pdac/filtered_pdac_cases_ids.tx"
SOURCE_DIR     = "/home/student1/ftzina_thesis/outputs/subregion_tumor_only/ALL_METRICS_0.4/HETEROGENEITY"          # folder with all 676 cases
OUTPUT_DIR     = "/home/student1/ftzina_thesis/output_pdac/intra_tumor_589_PDACs_0.4"      # new folder for 589 cases
#os.makedirs(OUTPUT_DIR, exist_ok=True)

SUFFIXES = [
    "_central_mask.npy",
    "_central_vector.npy",
    "_peripheral_mask.npy",
    "_peripheral_vector.npy",
    "_heterogeneity_vector.npy",
]

# Load valid case IDs
with open(VALID_IDS_FILE, "r") as f:
    valid_ids = set(line.strip() for line in f if line.strip())

print(f"Valid case IDs loaded: {len(valid_ids)}")


copied  = 0
missing = []

for case_id in sorted(valid_ids):
    for suffix in SUFFIXES:
        src = os.path.join(SOURCE_DIR, f"{case_id}{suffix}")
        dst = os.path.join(OUTPUT_DIR, f"{case_id}{suffix}")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(f"{case_id}{suffix}")

print(f"\nCopied : {copied} files")
print(f"Missing: {len(missing)} files")
if missing:
    print("Missing files:")
    for m in missing:
        print(f"  {m}")

# Verify: each case should have exactly 5 files
n_files = len(os.listdir(OUTPUT_DIR))
print(f"\nTotal files in output dir : {n_files}")
print(f"Expected                  : {len(valid_ids) * 5}")
assert n_files == len(valid_ids) * 5, "File count mismatch — check missing list above."
print("Verification passed.")
