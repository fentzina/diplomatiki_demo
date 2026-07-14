#!/usr/bin/env python3
"""
Isolates the PDAC-case CT files (listed by ID in pdac_ids.txt) out of all
4 batch folders into a single output folder.

- pdac_ids.txt: one case ID per line, in the SHORT form you have
  (e.g. "100000_00001"), NOT the filename form.
- Your actual files may be named either "<id>.nii.gz" or, more commonly for
  CT volumes, "<id>_0000.nii.gz" (the nnU-Net modality-suffix convention,
  e.g. "100000_00001_0000.nii.gz"). This script matches either form: it
  strips a trailing "_0000" off each filename's stem before comparing to
  your ID list, so both naming styles are handled automatically -- you
  don't need to touch pdac_ids.txt.
- Only ONE pass is made over the whole dataset (not one search per ID), so
  this stays fast even across 2240 files / 4 batches.

USAGE:
    python isolate_pdac_files.py \
        --ids_file pdac_ids.txt \
        --search_root /path/to/parent_dir_containing_all_4_batches \
        --out_dir PDAC_cases

    (If your batches are still zipped, extract them first -- this script
    only looks at files already on disk, it does not unzip anything.)

OUTPUT:
    - <out_dir>/                 <- the 678 matched files, copied here
    - missing_ids.txt            <- any IDs from pdac_ids.txt that were NOT
                                     found anywhere under --search_root
                                     (should be empty if everything matched)
"""

import os
import sys
import shutil
import argparse


def get_case_id(filename):
    """
    Strips .nii/.nii.gz, then strips a trailing '_0000' modality suffix if
    present, e.g. '100000_00001_0000.nii.gz' -> '100000_00001'.
    """
    if filename.endswith('.nii.gz'):
        stem = filename[:-7]
    elif filename.endswith('.nii'):
        stem = filename[:-4]
    else:
        return None  # not a NIfTI file

    if stem.endswith('_0000'):
        stem = stem[:-5]
    return stem


def load_ids(ids_file):
    ids = set()
    with open(ids_file, 'r') as f:
        for line in f:
            case_id = line.strip()
            if case_id:
                ids.add(case_id)
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Copy the CT files matching a list of case IDs (e.g. the 678 PDAC "
                    "cases) out of all batches into a single output folder."
    )
    parser.add_argument("--ids_file", type=str, default="pdac_ids.txt",
                         help="Text file with one case ID per line, short form "
                              "(e.g. '100000_00001'). Default: pdac_ids.txt")
    parser.add_argument("--search_root", type=str, required=True,
                         help="Parent directory containing all 4 batches' extracted "
                              "files (searched recursively).")
    parser.add_argument("--out_dir", type=str, default="PDAC_cases",
                         help="Where matched files are copied to. Default: PDAC_cases")
    parser.add_argument("--missing_log", type=str, default="missing_ids.txt",
                         help="Where unmatched IDs are logged. Default: missing_ids.txt")
    args = parser.parse_args()

    if not os.path.isfile(args.ids_file):
        print(f"ERROR: ID list not found at {args.ids_file}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load the wanted IDs into a set.
    wanted_ids = load_ids(args.ids_file)
    print(f"Loaded {len(wanted_ids)} target IDs from {args.ids_file}.")

    # 2. Single pass over every .nii / .nii.gz file under search_root.
    remaining_ids = set(wanted_ids)
    found = 0
    for dirpath, _dirnames, filenames in os.walk(args.search_root):
        for filename in filenames:
            if not (filename.endswith('.nii') or filename.endswith('.nii.gz')):
                continue

            case_id = get_case_id(filename)
            if case_id in remaining_ids:
                src = os.path.join(dirpath, filename)
                dst = os.path.join(args.out_dir, filename)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                remaining_ids.discard(case_id)
                found += 1

    # 3. Whatever's left in remaining_ids was never found.
    with open(args.missing_log, 'w') as f:
        for case_id in sorted(remaining_ids):
            f.write(case_id + "\n")

    print("----------------------------------------")
    print(f"Copied  : {found} / {len(wanted_ids)} files -> {args.out_dir}/")
    print(f"Missing : {len(remaining_ids)} (see {args.missing_log})")


if __name__ == "__main__":
    main()
