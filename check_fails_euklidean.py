import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Ορίστε το κεντρικό path των labels (το script θα ψάξει αυτόματα σε auto και manual)
LABEL_DIR = "/home/student1/ftzina_thesis/codes/panorama_labels"
# Το CSV report που παρήγαγες από το προηγούμενο βήμα
INPUT_CSV = "check_PWF_all.csv"
OUTPUT_CSV = "secondary_qc_report.csv"

DISTANCE_THRESHOLD_MM = 3.0  # Αυστηρό κλινικό όριο (3 χιλιοστά)
LABEL_SUFFIXES = ["_label.nii.gz", "_seg.nii.gz", "_mask.nii.gz", ".nii.gz"]

def find_case_path_recursive(label_dir, case_id):
    """Αναζήτηση του αρχείου NIfTI σε όλους τους υποφακέλους."""
    for root, dirs, files in os.walk(label_dir):
        for suffix in LABEL_SUFFIXES:
            target_file = f"{case_id}{suffix}"
            if target_file in files:
                return os.path.join(root, target_file)
    return None

def trustworthy_secondary_check(mask_path):
    """Υπολογισμός ελάχιστης Ευκλείδειας απόστασης (σε mm) μεταξύ όγκου και παγκρέατος."""
    img = sitk.ReadImage(mask_path)
    spacing = img.GetSpacing()  # (X, Y, Z) spacing σε mm
    mask_arr = sitk.GetArrayFromImage(img)  # (Z, Y, X) order
    
    tumor_mask = (mask_arr == 1)
    pancreas_mask = (mask_arr == 4)
    
    if not np.any(tumor_mask) or not np.any(pancreas_mask):
        return "Missing Labels", -1.0
    
    # Αντιστροφή μάσκας παγκρέατος (το πάγκρεας γίνεται 0, το υπόλοιπο 1)
    panc_inverse = ~pancreas_mask
    
    # Υπολογισμός χάρτη αποστάσεων λαμβάνοντας υπόψη το physical spacing (Z, Y, X order)
    distance_map = distance_transform_edt(panc_inverse, sampling=spacing[::-1])
    
    # Εύρεση της ελάχιστης απόστασης που έχει οποιοδήποτε voxel του όγκου από το πάγκρεας
    tumor_distances = distance_map[tumor_mask]
    min_distance_mm = float(np.min(tumor_distances))
    
    # Κατηγοριοποίηση
    if min_distance_mm <= DISTANCE_THRESHOLD_MM:
        verdict = "Exophytic Tumor (Keep)"
    else:
        verdict = "Technical Shift / Artifact (Exclude)"
        
    return verdict, round(min_distance_mm, 2)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Δεν βρέθηκε το αρχείο {INPUT_CSV}! Τρέξτε πρώτα το κεντρικό check_PWF.py .")
        return
        
    # Φόρτωση του προηγούμενου report
    df = pd.read_csv(INPUT_CSV, sep=";")
    
    # Απομόνωση ΜΟΝΟ των FAIL περιπτώσεων
    fail_df = df[df["verdict"] == "FAIL"].copy()
    total_fails = len(fail_df)
    print(f"Βρέθηκαν {total_fails} FAIL περιπτώσεις για δευτερογενή έλεγχο.")
    
    if total_fails == 0:
        print("Δεν υπάρχουν FAIL cases για έλεγχο!")
        return
        
    secondary_results = []
    
    # Τρέχουμε τον έλεγχο με progress bar
    for _, row in tqdm(fail_df.iterrows(), total=total_fails, desc="Running Secondary QC"):
        case_id = row["case_id"]
        mask_path = find_case_path_recursive(LABEL_DIR, case_id)
        
        if mask_path is None:
            secondary_results.append({
                "case_id": case_id,
                "secondary_verdict": "FILE NOT FOUND",
                "min_distance_mm": -1.0
            })
            continue
            
        try:
            verdict, dist_mm = trustworthy_secondary_check(mask_path)
            secondary_results.append({
                "case_id": case_id,
                "secondary_verdict": verdict,
                "min_distance_mm": dist_mm
            })
        except Exception as e:
            secondary_results.append({
                "case_id": case_id,
                "secondary_verdict": f"ERROR: {str(e)}",
                "min_distance_mm": -1.0
            })
            
    # Αποθήκευση των νέων αποτελεσμάτων σε CSV
    res_df = pd.DataFrame(secondary_results)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nΤο δευτερογενές report αποθηκεύτηκε στο: {OUTPUT_CSV}")
    
    # Εκτύπωση στατιστικών στο τερματικό
    print("\n" + "="*50)
    print("SECONDARY QC SUMMARY (FAIL CASES ANALYSIS)")
    print("="*50)
    print(res_df["secondary_verdict"].value_counts())
    print("="*50)

if __name__ == "__main__":
    main()
