#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil

# =================────────────────────────────────────────────────=============
# CONFIGURATION
# =================────────────────────────────────────────────────=============
# Παράδειγμα Linux διαδρομών: "/home/username/data/heterogeneity"
INPUT_DIR  = "/home/student1/ftzina_thesis/outputs/subregion_tumor_only/ALL_METRICS_0.4/HETEROGENEITY"
OUTPUT_DIR = "/home/student1/ftzina_thesis/output_pdac/intra_tumor_589_PDACs_0.4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Το όνομα του αρχείου .txt που περιέχει μόνο τα 589 έγκυρα IDs (ένα ανά γραμμή)
TXT_589_CASES = "filtered_pdac_cases_ids.txt"

# ==============================================================================
# 1. ΦΟΡΤΩΣΗ ΚΑΙ ΚΑΘΑΡΙΣΜΟΣ ΤΩΝ 589 ΕΓΚΥΡΩΝ IDS
# =================────────────────────────────────────────────────=============
print("Reading valid Scenario A case IDs...")
path_589 = os.path.join("/home/student1/ftzina_thesis/output_pdac", TXT_589_CASES)

with open(path_589, "r", encoding="utf-8") as f:
    # Το .replace('\r', '') αφαιρεί τους κρυφούς χαρακτήρες των Windows!
    ids_589 = [line.strip().replace('\r', '') for line in f if line.strip()]

print(f"  -> Target Scenario A pool loaded: {len(ids_589)} cases.")
if len(ids_589) > 0:
    print(f"  -> Sample parsed ID: '{ids_589[0]}' (Length: {len(ids_589[0])} chars)")

# ==============================================================================
# 2. ΑΠΟΜΟΝΩΣΗ ΚΑΙ ΑΝΤΙΓΡΑΦΗ ΤΩΝ .NPY ΑΡΧΕΙΩΝ ΑΝΑ ΑΣΘΕΝΗ
# =================────────────────────────────────────────────────=============
print("\nExecuting individual file isolation loop...")

suffixes = [
    "_central_mask.npy", 
    "_central_vector.npy", 
    "_heterogeneity_vector.npy", 
    "_peripheral_mask.npy", 
    "_peripheral_vector.npy"
]

copied_patients = 0
total_files_copied = 0

for pid in ids_589:
    patient_found = False
    
    for suffix in suffixes:
        filename = f"{pid}{suffix}"
        src_path = os.path.join(NPY_INPUT_DIR, filename)
        dst_path = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            total_files_copied += 1
            patient_found = True
        else:
            # Δείχνει ακριβώς πού έψαξε το αρχείο για να το επιβεβαιώσεις
            print(f"  [Warning] File NOT found: {src_path}")
            
    if patient_found:
        copied_patients += 1

# ==============================================================================
# 3. ΤΕΛΙΚΟΣ ΕΛΕΓΧΟΣ ΚΑΙ REPORT
# ==============================================================================
print(f"Pipeline complete! Successfully isolated {copied_patients} out of {len(ids_589)} valid patients.")
print(f"Total .npy files copied to clean directory: {total_files_copied} (Target: {len(ids_589) * 5})")
