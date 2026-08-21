import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Αλλάξτε το path με τη διαδρομή που έχετε το αρχείο μάσκας του συγκεκριμένου case
CASE_ID = "100196_00001"
LABEL_DIR = "/home/student1/ftzina_thesis/outputs/panorama_labels/automatic_labels" 

TUMOR_LABEL = 1
PANCREAS_LABEL = 4

def find_case_path(label_dir, case_id):
    suffixes = ["_label.nii.gz", "_seg.nii.gz", "_mask.nii.gz", ".nii.gz"]
    for s in suffixes:
        path = os.path.join(label_dir, f"{case_id}{s}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find mask for case {case_id} in {label_dir}")

def main():
    # 1. Φόρτωση της μάσκας
    mask_path = find_case_path(LABEL_DIR, CASE_ID)
    print(f"Loading mask from: {mask_path}")
    
    img = sitk.ReadImage(mask_path)
    mask_arr = sitk.GetArrayFromImage(img) # Shape: (Z, Y, X)
    
    # Διαχωρισμός των μασκών
    tumor_mask = (mask_arr == TUMOR_LABEL)
    pancreas_mask = (mask_arr == PANCREAS_LABEL)
    union_mask = tumor_mask | pancreas_mask
    
    # 2. Αυτόματος εντοπισμός του καλύτερου Slice (Κέντρο μάζας του όγκου)
    if not np.any(tumor_mask):
        raise ValueError(f"No tumor (label 1) found in case {CASE_ID}!")
        
    z_center, y_center, x_center = center_of_mass(tumor_mask)
    target_slice = int(round(z_center))
    print(f"Found tumor centered at Slice Z = {target_slice}")
    
    # Απόσπαση του 2D slice για το plot
    tumor_2d = tumor_mask[target_slice, :, :]
    pancreas_2d = pancreas_mask[target_slice, :, :]
    union_2d = union_mask[target_slice, :, :]
    
    # 3. Δημιουργία RGB Overlays για να φαίνονται όμορφα και καθαρά
    # Panel 1: Donut Effect (Μπλε Πάγκρεας, Κόκκινος Όγκος)
    donut_rgb = np.zeros((tumor_2d.shape[0], tumor_2d.shape[1], 3), dtype=np.uint8)
    donut_rgb[pancreas_2d] = [41, 128, 185]  # Ωραίο Μπλε
    donut_rgb[tumor_2d] = [231, 76, 60]     # Έντονο Κόκκινο
    
    # Panel 2: Union Mask (Ενιαίο Πράσινο ROI)
    union_rgb = np.zeros((union_2d.shape[0], union_2d.shape[1], 3), dtype=np.uint8)
    union_rgb[union_2d] = [46, 204, 113]     # Ωραίο Πράσινο (Whole Organ)
    
    # 4. Αυτόματο Κροπάρισμα (Zoom-in) γύρω από το όργανο για να μη φαίνεται χαμένο στο μαύρο φόντο
    y_indices, x_indices = np.where(union_2d)
    y_min, y_max = y_indices.min() - 20, y_indices.max() + 20
    x_min, x_max = x_indices.min() - 20, x_indices.max() + 20
    
    # Διασφάλιση ορίων πίνακα
    y_min, y_max = max(0, y_min), min(tumor_2d.shape[0], y_max)
    x_min, x_max = max(0, x_min), min(tumor_2d.shape[1], x_max)
    
    # 5. Σχεδιασμός με το Matplotlib (2-Panel Figure)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor='#1e1e1e')
    
    # --- Panel 1: The Problem ---
    axes[0].imshow(donut_rgb[y_min:y_max, x_min:x_max])
    axes[0].set_title(f"Image 1: The Problem (Donut Effect)\nCase: {CASE_ID} | Slice Z: {target_slice}", 
                      color='white', fontsize=14, pad=15, fontweight='bold')
    axes[0].axis('off')
    # Προσθήκη κειμένου/λεζάντας πάνω στο plot
    axes[0].text(0.05, 0.05, "Blue: Pancreas Parenchyma\nRed: PDAC Tumor\n Voxel Overlap = 0.0%", 
                 transform=axes[0].transAxes, color='white', fontsize=11,
                 bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # --- Panel 2: The Solution ---
    axes[1].imshow(union_rgb[y_min:y_max, x_min:x_max])
    axes[1].set_title("Image 2: Our Solution (Whole Organ ROI)\nUnion Mask (Label 1 + Label 4)", 
                      color='white', fontsize=14, pad=15, fontweight='bold')
    axes[1].axis('off')
    axes[1].text(0.05, 0.05, "Green: Combined ROI\nRestores Anatomical Whole\n Ready for Radiomics", 
                 transform=axes[1].transAxes, color='white', fontsize=11,
                 bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Αποθήκευση της τελικής εικόνας
    output_filename = f"donut_effect_evidence_{CASE_ID}.png"
    plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Success! Evidence image saved as: {output_filename}")

if __name__ == "__main__":
    main()
