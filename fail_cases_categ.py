import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_dilation

def categorize_fail_case(image_path, label_path):
    """
    Κατηγοριοποιεί μια FAIL περίπτωση σε Κατηγορία Α (Exophytic) ή Κατηγορία Β (Technical Shift).
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
        
    # 1. Κάνουμε Dilation (επέκταση) στον όγκο κατά 2 voxels για να ελέγξουμε την επαφή (3D αλληλεπίδραση)
    dilated_tumor = binary_dilation(tumor_mask, iterations=2)
    
    # Έλεγχος αν ο διευρυμένος όγκος ακουμπάει το πάγκρεας
    has_3d_contact = np.any(dilated_tumor & pancreas_mask)
    
    # 2. Υπολογισμός των slices στα οποία εμφανίζεται το κάθε label (Διορθωμένο συντακτικό)
    tumor_slices = np.where(np.any(tumor_mask, axis=(0, 1)))[0]
    pancreas_slices = np.where(np.any(pancreas_mask, axis=(0, 1)))[0]
    
    # Εύρεση των κοινών slices στα οποία συνυπάρχουν
    common_slices = np.intersect1d(tumor_slices, pancreas_slices)
    
    # 3. Λογική Απόφασης (Decision Logic)
    if has_3d_contact:
        verdict = "CATEGORY_A"
        details = f"Exophytic. Contact verified. Tumor active in {len(tumor_slices)} slices. Overlap in {len(common_slices)} slices."
    else:
        verdict = "CATEGORY_B"
        details = "Technical Shift. Tumor and Pancreas are completely isolated in 3D space."
        
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
    LABEL_DIRECTORY = "/home/student1/ftzina_thesis/codes/panorama_labels/automatic_labels"
    
    fail_cases = ['100003_00001', '100005_00001', '100011_00001', '100016_00001', '100021_00001', '100033_00001', '100082_00001',
 '100091_00001', '100096_00001', '100126_00001', '100135_00001', '100143_00001', '100150_00001', '100157_00001', '100164_00001', '100180_00001', '100196_00001',
 '100197_00001', '100200_00001', '100213_00001', '100219_00001', '100229_00001', '100243_00001', '100248_00001', '100257_00001', '100262_00001',
 '100265_00001', '100265_00002', '100266_00001', '100285_00001', '100328_00001', '100345_00001', '100355_00001', '100356_00001',
 '100372_00001', '100373_00001', '100381_00001', '100386_00001', '100394_00001', '100407_00001', '100413_00001', '100426_00001', '100436_00001', '100462_00001', '100467_00001', '100474_00001', '100477_00001', '100512_00001', '100520_00001',
 '100528_00001', '100554_00001', '100574_00001', '100616_00001', '100619_00001', '100636_00001', '100642_00001', '100647_00001', '100703_00001', '100709_00001', '100740_00001', '100745_00001', '100747_00001', '100757_00001',
 '100822_00001', '100826_00001', '100842_00001', '100850_00001', '100852_00001', '100871_00001', '100903_00001', '100904_00001', '100905_00001', '100908_00001', '100926_00001', '100933_00001',
 '100934_00001', '100943_00001', '100948_00001', '100950_00001', '100956_00001', '100959_00001', '100983_00001', '100988_00001',
 '100994_00001', '101013_00001', '101014_00001', '101026_00001', '101038_00001', '101043_00001', '101048_00001', '101063_00001', '101068_00001', '101102_00001',
 '101112_00001', '101118_00001', '101120_00001', '101137_00001', '101142_00001', '101144_00001', '101160_00001', '101161_00001', '101169_00001', '101175_00001', '101196_00001', '101205_00001', '101227_00001', '101229_00001',
 '101249_00001', '101254_00001', '101264_00001', '101267_00001', '101322_00001', '101331_00001', '101336_00001', '101339_00001', '101345_00001', '101355_00001', '101367_00001',
 '101375_00001', '101376_00001', '101377_00001', '101396_00001', '101408_00001', '101417_00001', '101418_00001', '101425_00001', '101427_00001', '101436_00001',
 '101488_00001', '101503_00001', '101533_00001', '101534_00001', '101543_00001', '101545_00001', '101550_00001', '101560_00001', '101563_00001', '101593_00001', '101596_00001', '101616_00001','101625_00001',
 '101634_00001', '101637_00001', '101639_00001', '101658_00001', '101692_00001', '101696_00001', '101698_00001', '101717_00001', '101735_00001', '101744_00001', '101767_00001',
 '101787_00001', '101794_00001', '101795_00001', '101808_00001', '101813_00001', '101829_00001', '101836_00001', '101845_00001', '101855_00001', '101906_00001', '101912_00001',
 '101918_00001', '101930_00001', '101956_00001', '101964_00001', '101980_00001', '101988_00001', '102009_00001', '102020_00001', '102039_00001', '102040_00001', '102041_00001', '102044_00001', '102057_00001', '102065_00001', '102065_00002', '102066_00001',
 '102072_00001', '102090_00001', '102105_00001', '102111_00001', '102133_00001', '102137_00001', '102138_00001','102140_00001',
 '102147_00001', '102160_00001', '102164_00001', '102167_00001', '102190_00001', '102195_00001', '102203_00001', '102204_00001',
 '102216_00001', '102218_00001']
    
    run_qc2_analysis(LABEL_DIRECTORY, fail_cases, "qc2_fail_analysis_report.csv")
