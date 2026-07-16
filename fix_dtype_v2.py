import os
import sys
import numpy as np
from tqdm import tqdm

# ---- EDIT THIS PATH IF NEEDED ----
tensor_dir = os.path.expanduser("~/ftzina_thesis/outputs/ALL_TENSORS_1/npy_files")
# -----------------------------------

tensor_dir = os.path.abspath(tensor_dir)
print(f"Resolved path : {tensor_dir}")
print(f"Path exists?  : {os.path.exists(tensor_dir)}")
print(f"Is directory? : {os.path.isdir(tensor_dir)}")

if not os.path.isdir(tensor_dir):
    print("ERROR: tensor_dir is not a valid directory. Fix the path above and rerun.")
    sys.exit(1)

# Recursive search — finds .npy files even in nested subfolders
npy_files = []
for root, dirs, files in os.walk(tensor_dir):
    for f in files:
        if f.lower().endswith('.npy'):
            npy_files.append(os.path.join(root, f))
npy_files.sort()

print(f"Found {len(npy_files)} .npy files (recursive search).")

if len(npy_files) == 0:
    print("Still zero — printing raw directory listing for debugging:")
    print(os.listdir(tensor_dir)[:20])
    sys.exit(1)

converted, skipped = 0, 0
pbar = tqdm(npy_files, unit="file")
for fpath in pbar:
    pbar.set_description(os.path.basename(fpath))
    t = np.load(fpath, mmap_mode='r')
    if t.dtype != np.float32:
        t32 = t.astype(np.float32)
        del t  # release the mmap before touching the file on disk

        tmp_path = fpath + '.tmp'
        np.save(tmp_path, t32)
        os.replace(tmp_path, fpath)  # atomic on POSIX
        converted += 1
    else:
        skipped += 1

print(f"\nDone. Converted: {converted}, already float32: {skipped}")
