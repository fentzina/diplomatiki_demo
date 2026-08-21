import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Το ακριβές μονοπάτι του αρχείου NIfTI που μου έδωσες
MASK_PATH = "/home/student1/ftzina_thesis/codes/panorama_labels/automatic_labels/100196_00001.nii.gz"
CASE_ID = "100196_00001"

TUMOR_LABEL = 1
PANCREAS_LABEL = 4

def main():
    print(f"Loading mask directly from: {MASK_PATH}")
    
    # 1. Φόρτωση της μάσκας
    img = sitk.ReadImage(MASK_PATH)
    mask_arr = sitk.GetArrayFromImage(img) # Shape: (Z, Y, X)
    
    # Διαχωρισμός των μασκών
    tumor_mask = (mask_arr == TUMOR_LABEL)
    pancreas_mask = (mask_arr == PANCREAS_LABEL)
    union_mask = tumor_mask | pancreas_mask
    
    # 2. Αυτόματος εντοπισμός του καλύτερου Slice (Κέντρο μάζας του όγκου)
    if not np.any(tumor_mask):
        print(f"Σφάλμα: Δεν βρέθηκε όγκος (label 1) στο αρχείο!")
        return
        
    z_center, y_center, x_center = center_of_mass(tumor_mask)
    target_slice = int(round(z_center))
    print(f"Ο όγκος εντοπίστηκε στο Slice Z = {target_slice}")
    
    # Απόσπαση του 2D slice για το plot
    tumor_2d = tumor_mask[target_slice, :, :]
    pancreas_2d = pancreas_mask[target_slice, :, :]
    union_2d = union_mask[target_slice, :, :]
    
    # 3. Δημιουργία RGB Overlays
    # Panel 1: Donut Effect (Μπλε Πάγκρεας, Κόκκινος Όγκος)
    donut_rgb = np.zeros((tumor_2d.shape[0], tumor_2d.shape[1], 3), dtype=np.uint8)
    donut_rgb[pancreas_2d] = [0, 120, 255]  # Ωραίο Μπλε
    donut_rgb[tumor_2d] = [255, 40, 40]     # Έντονο Κόκκινο
    
    # Panel 2: Union Mask (Ενιαίο Πράσινο ROI)
    union_rgb = np.zeros((union_2d.shape[0], union_2d.shape[1], 3), dtype=np.uint8)
    union_rgb[union_2d] = [40, 200, 100]    # Ωραίο Πράσινο (Whole Organ)
    
    # 4. Αυτόματο Κροπάρισμα (Zoom-in) γύρω από το όργανο
    y_indices, x_indices = np.where(union_2d)
    if len(y_indices) > 0:
        y_min, y_max = y_indices.min() - 20, y_indices.max() + 20
        x_min, x_max = x_indices.min() - 20, x_indices.max() + 20
        y_min, y_max = max(0, y_min), min(tumor_2d.shape[0], y_max)
        x_min, x_max = max(0, x_min), min(tumor_2d.shape[1], x_max)
    else:
        y_min, y_max, x_min, x_max = 0, tumor_2d.shape[0], 0, tumor_2d.shape[1]
    
    # 5. Σχεδιασμός με το Matplotlib (2-Panel Figure)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='#1e1e1e')
    
    # --- Panel 1: The Problem ---
    axes[0].imshow(donut_rgb[y_min:y_max, x_min:x_max])
    axes[0].set_title(f"Image 1: The Problem (Donut Effect)\nCase: {CASE_ID} | Slice Z: {target_slice}", 
                      color='white', fontsize=14, pad=15, fontweight='bold')
    axes[0].axis('off')
    axes[0].text(0.05, 0.05, "Blue: Pancreas Parenchyma\nRed: PDAC Tumor\nVoxel Overlap = 0.0%", 
                 transform=axes[0].transAxes, color='white', fontsize=11,
                 bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # --- Panel 2: The Solution ---
    axes[1].imshow(union_rgb[y_min:y_max, x_min:x_max])
    axes[1].set_title("Image 2: Our Solution (Whole Organ ROI)\nUnion Mask (Label 1 + Label 4)", 
                      color='white', fontsize=14, pad=15, fontweight='bold')
    axes[1].axis('off')
    axes[1].text(0.05, 0.05, "Green: Combined ROI\nRestores Anatomical Whole\nReady for Radiomics", 
                 transform=axes[1].transAxes, color='white', fontsize=11,
                 bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Αποθήκευση της τελικής εικόνας
    output_filename = f"donut_effect_evidence_{CASE_ID}.png"
    plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Η εικόνα αποθηκεύτηκε επιτυχώς ως: {output_filename}")

if __name__ == "__main__":
    main()
