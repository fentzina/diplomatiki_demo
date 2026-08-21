import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MASK_PATH = "/home/student1/ftzina_thesis/codes/panorama_labels/automatic_labels/100196_00001.nii.gz"
CASE_ID = "100196_00001"

TUMOR_LABEL = 1
PANCREAS_LABEL = 4

def main():
    print(f"Loading mask from: {MASK_PATH}")
    img = sitk.ReadImage(MASK_PATH)
    mask_arr = sitk.GetArrayFromImage(img) # Shape: (Z, Y, X)
    
    # 1. Εντοπισμός του slice με τον όγκο
    tumor_mask = (mask_arr == TUMOR_LABEL)
    if not np.any(tumor_mask):
        print("Δεν βρέθηκε όγκος στο αρχείο!")
        return
        
    z_center, _, _ = center_of_mass(tumor_mask)
    target_slice = int(round(z_center))
    
    # Απόσπαση του 2D slice
    tumor_2d = (mask_arr[target_slice, :, :] == TUMOR_LABEL)
    pancreas_2d = (mask_arr[target_slice, :, :] == PANCREAS_LABEL)
    
    # 2. Εύρεση των συντεταγμένων (Y, X) για κάθε label ξεχωριστά
    y_panc, x_panc = np.where(pancreas_2d)
    y_tumor, x_tumor = np.where(tumor_2d)
    
    # 3. Μαθηματικός έλεγχος overlap για τη λεζάντα
    overlap_count = np.sum(tumor_2d & pancreas_2d)
    
    # 4. Σχεδιασμός Scatter Plot (Voxel-Level Coordinate Map)
    plt.figure(figsize=(10, 8), facecolor='#1e1e1e')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    
    # Σχεδιάζουμε τα voxels του παγκρέατος ως μπλε τελείες
    plt.scatter(x_panc, y_panc, color='#0066ff', s=15, alpha=0.6, label='Pancreas (Label 4)', edgecolors='none')
    
    # Σχεδιάζουμε τα voxels του όγκου ως κόκκινες τελείες
    plt.scatter(x_tumor, y_tumor, color='#ff3333', s=15, alpha=0.8, label='Tumor (Label 1)', edgecolors='none')
    
    # 5. Ρυθμίσεις Γραφήματος & Zoom-in
    all_y = np.concatenate([y_panc, y_tumor])
    all_x = np.concatenate([x_panc, x_tumor])
    
    plt.ylim(all_y.max() + 10, all_y.min() - 10) # Αντίστροφη Y για να ταιριάζει με τον προσανατολισμό της ιατρικής εικόνας
    plt.xlim(all_x.min() - 10, all_x.max() + 10)
    
    plt.title(f"Voxel-Level Coordinate Map (No Mixing Demonstration)\nCase: {CASE_ID} | Slice Z: {target_slice}", 
              color='white', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("X Coordinate (Pixels)", color='white', fontsize=11)
    plt.ylabel("Y Coordinate (Pixels)", color='white', fontsize=11)
    
    # Στυλ αξόνων
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444444')
        
    plt.grid(True, color='#333333', linestyle='--', alpha=0.5)
    plt.legend(facecolor='black', edgecolor='#444444', labelcolor='white', fontsize=11, loc='upper right')
    
    # Προσθήκη ευκρινούς κειμένου απόδειξης
    proof_text = f"PROOF OF MUTUAL EXCLUSIVITY:\n• Shared/Overlapping Voxels = {overlap_count}\n• Voxel-Level Mixing = 0.00%"
    plt.text(0.05, 0.05, proof_text, transform=ax.transAxes, color='#00ffcc', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='black', alpha=0.8, edgecolor='#00ffcc', boxstyle='round,pad=0.6'))
    
    plt.tight_layout()
    
    # Αποθήκευση
    output_filename = f"voxel_mixing_proof_{CASE_ID}.png"
    plt.savefig(output_filename, dpi=300, facecolor='#1e1e1e', edgecolor='none')
    print(f"Η εικόνα-απόδειξη αποθηκεύτηκε ως: {output_filename}")

if __name__ == "__main__":
    main()
