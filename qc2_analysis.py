import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_dilation

def categorize_fail_case(image_path, label_path):
    """
    Κατηγοριοποιεί μια FAIL περίπτωση σε Κατηγορία Α (Exophytic/Nested) ή Κατηγορία Β (Technical Shift).
    """
    # Φόρτωση του NIfTI αρχείου των labels
    img = nib.load(label_path)
    data = img.get_fdata()
    
    # Διαχωρισμός των 3D πινάκων για Tumor (1) και Pancreas (4)
    tumor_mask = (data == 1)
    pancreas_mask = (data == 4)
    
    total_tumor_voxels = np.sum(tumor_mask)
    if total_tumor_voxels == 0:
        return "SKIP", 0, "No tumor voxels found"
        
    # 1. Αυξημένο Dilation (επέκταση) στον όγκο κατά 5 voxels για απόλυτη ασφάλεια
    # Γεφυρώνει τεχνητά κενά (margins) γύρω από μικρούς, σωστά τοποθετημένους όγκους
    dilated_tumor = binary_dilation(tumor_mask, iterations=5)
    
    # Έλεγχος αν ο διευρυμένος όγκος ακουμπάει το πάγκρεας σε 3D επίπεδο
    has_3d_contact = np.any(dilated_tumor & pancreas_mask)
    
    # 2. Υπολογισμός των slices στα οποία εμφανίζεται το κάθε label (Διορθωμένο)
    tumor_slices = np.where(np.any(tumor_mask, axis=(0, 1)))[0]
    pancreas_slices = np.where(np.any(pancreas_mask, axis=(0, 1)))[0]
    
    # Εύρεση των κοινών slices στα οποία συνυπάρχουν κατά μήκος του Z-άξονα
    common_slices = np.intersect1d(tumor_slices, pancreas_slices)
    
    # 3. Λογική Απόφασης (100% Ασφαλής για True Category B)
    # Μια περίπτωση βγαίνει Category B ΜΟΝΟ αν δεν έχει 3D επαφή ΚΑΙ δεν μοιράζεται κοινά slices
    if has_3d_contact or len(common_slices) > 0:
        verdict = "CATEGORY_A"
        details = f"Anatomically Valid. 3D Contact: {has_3d_contact}. Tumor active in {len(tumor_slices)} slices. Z-axis overlap in {len(common_slices)} slices."
    else:
        verdict = "CATEGORY_B"
        details = "True Technical Shift. Tumor and Pancreas are completely isolated in 3D space and different slices."
        
    return verdict, total_tumor_voxels, details

def run_qc2_analysis(label_dir, fail_cases_list, output_csv):
    results = []
    
    print(f"Starting QC2 Fail Categorization for {len(fail_cases_list)} cases...")
    
    for i, case_id in enumerate(fail_cases_list, 1):
        filename = f"{case_id}.nii.gz" 
        label_path = os.path.join(label_dir, filename)
        
        print(f"[{i}/{len(fail_cases_list)}] Processing: {case_id}", end="\r")
        
        if not os.path.exists(label_path):
            results.append({"case_id": case_id, "verdict": "NOT_FOUND", "tumor_voxels": 0, "details": "File missing"})
            continue
            
        try:
            verdict, voxels, details = categorize_fail_case(None, label_path)
            results.append({
                "case_id": case_id,
                "verdict": verdict,
                "tumor_voxels": voxels,
                "details": details
            })
        except Exception as e:
            results.append({"case_id": case_id, "verdict": "ERROR", "tumor_voxels": 0, "details": str(e)})

    # Αποθήκευση σε νέο CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Εκτύπωση Στατιστικών
    print("\n\n=== QC2 CATEGORIZATION SUMMARY ===")
    print(df['verdict'].value_counts())
    print(f"Report saved to: {output_csv}")

if __name__ == "__main__":
    # Έχετε επιλέξει τον φάκελο manual_labels
    LABEL_DIRECTORY = "/home/student1/ftzina_thesis/codes/panorama_labels/manual_labels"
    
    fail_cases = [
        '100134_00001', '100705_00001', '101900_00001', '100936_00001', '102139_00001', 
        '102051_00001', '100113_00001', '101478_00001', '101773_00001', '101391_00001', '101083_00001',
        '101517_00001', '101994_00001', '100235_00001', '100967_00001', '102206_00001', '101584_00001', 
        '100065_00001', '101151_00001', '102084_00001', '101872_00001', '102130_00001', '100725_00001',
        '100086_00001', '101290_00001', '101085_00001', '102054_00001', '100405_00001', '101694_00001', 
        '101360_00001', '100419_00001', '101600_00001', '101917_00001', '102061_00001', '101031_00001', 
        '101023_00001', '101314_00001', '101178_00001', '100397_00001', '101163_00001', '101133_00001', 
        '101604_00001', '102005_00001', '100074_00001', '100733_00001', '100803_00001', '101378_00001', 
        '100316_00001', '100379_00001', '100438_00001', '100526_00001', '100043_00001', '101564_00001', 
        '101321_00001', '100664_00001', '100427_00001'
    ]  

    run_qc2_analysis(LABEL_DIRECTORY, fail_cases, "qc2_fail_analysis_report_manual.csv")
