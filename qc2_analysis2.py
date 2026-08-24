import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import center_of_mass

def categorize_fail_case_with_distance(label_path, distance_threshold_mm=50.0):
    """
    Κατηγοριοποιεί με βάση την 3D Ευκλείδεια απόσταση των κέντρων μάζας (Centroids) σε χιλιοστά.
    """
    img = nib.load(label_path)
    data = img.get_fdata()
    
    # Λήψη του voxel size (spacing) σε mm από το header (π.χ. [0.7, 0.7, 1.0])
    header = img.header
    zooms = header.get_zooms() # (dx, dy, dz)
    
    tumor_mask = (data == 1)
    pancreas_mask = (data == 4)
    
    total_tumor_voxels = np.sum(tumor_mask)
    if total_tumor_voxels == 0:
        return "SKIP", 0, 0.0, "No tumor voxels found"
        
    # 1. Υπολογισμός των Κέντρων Μάζας (Indices στον πίνακα)
    tumor_centroid_idx = center_of_mass(tumor_mask)
    pancreas_centroid_idx = center_of_mass(pancreas_mask)
    
    # 2. Μετατροπή των συντεταγμένων από pixels σε χιλιοστά (mm)
    tumor_centroid_mm = np.array(tumor_centroid_idx) * np.array(zooms)
    pancreas_centroid_mm = np.array(pancreas_centroid_idx) * np.array(zooms)
    
    # 3. Υπολογισμός 3D Ευκλείδειας Απόστασης
    distance_mm = np.linalg.norm(tumor_centroid_mm - pancreas_centroid_mm)
    
    # 4. Λογική Απόφασης (Βάσει ιατρικής απόστασης)
    if distance_mm > distance_threshold_mm:
        # Αν απέχουν πάνω από 5 εκατοστά, είναι 100% True Technical Shift (Category B)
        verdict = "CATEGORY_B"
        details = f"True Technical Shift. Centroids distance is {distance_mm:.2f} mm (> {distance_threshold_mm} mm)."
    else:
        # Αν είναι κοντά, είναι ανατομικά έγκυρος όγκος (Category A)
        verdict = "CATEGORY_A"
        details = f"Anatomically Valid. Centroids distance is {distance_mm:.2f} mm (≤ {distance_threshold_mm} mm)."
        
    return verdict, total_tumor_voxels, distance_mm, details

def run_qc2_analysis_with_distance(label_dir, fail_cases_list, output_csv):
    results = []
    print(f"Starting Distance-based QC2 Categorization for {len(fail_cases_list)} cases...")
    
    for i, case_id in enumerate(fail_cases_list, 1):
        filename = f"{case_id}.nii.gz" 
        label_path = os.path.join(label_dir, filename)
        
        print(f"[{i}/{len(fail_cases_list)}] Processing: {case_id}", end="\r")
        
        if not os.path.exists(label_path):
            results.append({"case_id": case_id, "verdict": "NOT_FOUND", "distance_mm": 0.0, "details": "File missing"})
            continue
            
        try:
            verdict, voxels, dist, details = categorize_fail_case_with_distance(label_path)
            results.append({
                "case_id": case_id,
                "verdict": verdict,
                "distance_mm": round(dist, 2),
                "details": details
            })
        except Exception as e:
            results.append({"case_id": case_id, "verdict": "ERROR", "distance_mm": 0.0, "details": str(e)})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print("\n\n=== QC2 DISTANCE-BASED SUMMARY ===")
    print(df['verdict'].value_counts())
    print(f"Report saved to: {output_csv}")

if __name__ == "__main__":
    LABEL_DIRECTORY = "/home/student1/ftzina_thesis/codes/panorama_labels/manual_labels"
    
    fail_cases = [
        '100082_00001', '100134_00001', '100705_00001', '101900_00001', '100936_00001', '102139_00001', 
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

    run_qc2_analysis_with_distance(LABEL_DIRECTORY, fail_cases, "qc2_fail_analysis_report_manual_new.csv")
