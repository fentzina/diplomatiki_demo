#!/usr/bin/env python3
"""
Automated 3D CT Preprocessing & CoLIAGe Feature Extraction Pipeline
Usage:
    python preprocess_batch.py --ct_dir /path/to/ct_images --label_dir /path/to/labels --output_dir /path/to/output
"""

import os
import sys
import glob
import math
import logging
import argparse
from itertools import product
from enum import Enum, IntEnum

import numpy as np
import SimpleITK as sitk
import nibabel as nib
import scipy.ndimage as ndi
import mahotas as mt
from scipy import linalg
from skimage.feature import graycomatrix
from skimage.util.shape import view_as_windows

import os
import glob
import zipfile

# ── 1. Ρύθμιση Μονοπατιών για τον Server ──────────────────────────────────────
BASE_DIR = os.environ.get('COLLAGE_BASE_DIR', '/home/student1/ftzina_thesis')

# Ο φάκελος 'data' περιέχει τα κατεβασμένα ZIP (π.χ. batch_4.zip ή batch_1.zip)
# Αλλάξτε το 'batch_4.zip' ανάλογα με το ποιο ZIP περιέχει το αρχείο δοκιμής σας
ZIP_NAME = 'batch_1.zip' 
ZIP_PATH = os.path.join(BASE_DIR, 'data', ZIP_NAME)

DATA_ROOT = os.path.join(BASE_DIR, 'panorama_cases')
os.makedirs(DATA_ROOT, exist_ok=True)

# ── 2. Στοχευμένη Αποσυμπίεση ΜΟΝΟ του Αρχείου Δοκιμής ────────────────────────
TARGET_FILE = "100029_00001_0000.nii.gz"
extracted_file_path = os.path.join(DATA_ROOT, TARGET_FILE)

if os.path.exists(extracted_file_path):
    print(f"[TEST] Το αρχείο δοκιμής {TARGET_FILE} υπάρχει ήδη στο {DATA_ROOT} — skipping extraction.")
else:
    if not os.path.isfile(ZIP_PATH):
        raise FileNotFoundError(f"Το ZIP αρχείο δεν βρέθηκε στη διαδρομή: '{ZIP_PATH}'")
    
    print(f"[TEST] Αναζήτηση και εξαγωγή του {TARGET_FILE} από το {ZIP_NAME}...")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        # Ψάχνουμε το αρχείο μέσα στο ZIP (μπορεί να είναι μέσα σε υποφάκελο)
        target_internal_path = None
        for member in zf.namelist():
            if member.endswith(TARGET_FILE):
                target_internal_path = member
                break
        
        if target_internal_path is None:
            raise FileNotFoundError(f"Το αρχείο {TARGET_FILE} δεν βρέθηκε μέσα στο {ZIP_NAME}!")
        
        # Εξαγωγή μόνο αυτού του αρχείου
        zf.extract(target_internal_path, DATA_ROOT)
        
        # Αν το zipfile το έβγαλε μέσα σε υποφάκελο, το μεταφέρουμε χύμα στο DATA_ROOT
        actual_extracted_path = os.path.join(DATA_ROOT, target_internal_path)
        if actual_extracted_path != extracted_file_path:
            shutil.move(actual_extracted_path, extracted_file_path)
            
    print(f"[TEST] Το αρχείο {TARGET_FILE} είναι έτοιμο στη διαδρομή: {extracted_file_path}")

# ── 3. Δημιουργία Λίστας Αρχείων ΑΠΟΚΛΕΙΣΤΙΚΑ για τη Δοκιμή ──────────────────
# Αντί για glob σε όλο τον φάκελο, βάζουμε καρφωτά μόνο το αρχείο δοκιμής
all_ct_files = [extracted_file_path]
print(f"\n[TEST MODE] Έτοιμο για δοκιμή με {len(all_ct_files)} αρχείο CT.")


# ---- 2. Build the full, sorted list of CT case files across all 4 batches --
all_ct_files = sorted(
    glob.glob(os.path.join(DATA_ROOT, '**', '*.nii.gz'), recursive=True)
)
print(f"\nFound {len(all_ct_files)} total CT case files across all batches.")
assert len(all_ct_files) > 0, "No .nii.gz files found — check ZENODO_BATCH_URLS / zip structure."

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocess_pipeline')

# --- CONSTANTS ---
TARGET_SHAPE = (128, 128, 128)   # cubic crop
HU_MIN, HU_MAX = -100, 600       # pancreas window
TUMOR_LABEL = 1
PANCREAS_LABEL = 4

# ==========================================
# 1. COLIAGE IMPLEMENTATION
# ==========================================

def _svd_dominant_angles(dx, dy, dz, svd_radius):
    is_3D = dx.shape[2] > 1
    svd_diameter = svd_radius * 2 + 1
    window_shape = (svd_diameter, svd_diameter, 1)
    
    dx_windows = view_as_windows(dx, window_shape)
    dy_windows = view_as_windows(dy, window_shape)
    dz_windows = view_as_windows(dz, window_shape)

    angles_shape = dx_windows.shape[0:3]
    dominant_angles_array = np.zeros(angles_shape + (2 if is_3D else 1,), np.single)

    center_x_range = range(angles_shape[1])
    center_y_range = range(angles_shape[0])
    center_z_range = range(angles_shape[2])
    for x, y, z in product(center_x_range, center_y_range, center_z_range):
        dominant_angles_array[y, x, z, :] = _svd_dominant_angle(x, y, z, dx_windows, dy_windows, dz_windows)

    return dominant_angles_array


def _svd_dominant_angle(x, y, z, dx_windows, dy_windows, dz_windows):
    dx_patch = dx_windows[y, x, z]
    dy_patch = dy_windows[y, x, z]
    dz_patch = dz_windows[y, x, z]

    is_3D = dx_windows.shape[2] > 1
    window_area = dx_patch.size
    flattened_gradients = np.zeros((window_area, (3 if is_3D else 2)))
    matrix_order = 'F'
    flattened_gradients[:, 0] = np.reshape(dx_patch, window_area, order=matrix_order)
    flattened_gradients[:, 1] = np.reshape(dy_patch, window_area, order=matrix_order)
    if is_3D:
        flattened_gradients[:, 2] = np.reshape(dz_patch, window_area, order=matrix_order)

    _, _, v = linalg.svd(flattened_gradients)

    dominant_y = v[0, 0]
    dominant_x = v[1, 0]
    dominant_angle = math.atan2(dominant_y, dominant_x)

    if is_3D:
        dominant_z = v[2, 0]
        secondary_angle = math.atan2(dominant_z, math.sqrt(dominant_x ** 2 + dominant_y ** 2))
        return (dominant_angle, secondary_angle)
    else:
        return dominant_angle


class DifferenceVarianceInterpretation(Enum):
    XMinusYVariance = 0
    ProbabilityXMinusYVariance = 1


class Collage:
    def __init__(self, img_array, mask_array, svd_radius=5, cooccurence_angles=[x * np.pi/4 for x in range(8)], num_unique_angles=64):
        self._haralick_window_size = svd_radius * 2 + 1
        self._svd_radius = svd_radius
        self._num_unique_angles = num_unique_angles
        self._cooccurence_angles = cooccurence_angles
        self._is_3D = img_array.ndim == 3
        
        self._img_array = img_array if self._is_3D else img_array.reshape(img_array.shape + (1,))
        thresholded_mask_array = (mask_array != 0).reshape(self._img_array.shape)

        non_zero_indices = np.argwhere(thresholded_mask_array)
        min_mask_coordinates = non_zero_indices.min(0)
        max_mask_coordinates = non_zero_indices.max(0) + 1
        
        self.mask_min_x, self.mask_min_y, self.mask_min_z = min_mask_coordinates[1], min_mask_coordinates[0], min_mask_coordinates[2]
        self.mask_max_x, self.mask_max_y, self.mask_max_z = max_mask_coordinates[1], max_mask_coordinates[0], max_mask_coordinates[2]

        self._mask_array = thresholded_mask_array[self.mask_min_y:self.mask_max_y, self.mask_min_x:self.mask_max_x, self.mask_min_z:self.mask_max_z]
        self.collage_output = None

    def _calculate_haralick_feature_values(self, img_array, center_x, center_y):
        window_size = self._haralick_window_size
        min_x = int(max(0, center_x - window_size / 2 - 1))
        min_y = int(max(0, center_y - window_size / 2 - 1))
        max_x = int(min(img_array.shape[1] - 1, center_x + window_size / 2 + 1))
        max_y = int(min(img_array.shape[0] - 1, center_y + window_size / 2 + 1))
        cropped_img_array = img_array[min_y:max_y, min_x:max_x]

        cooccurence_matrix = graycomatrix(cropped_img_array, [1], self._cooccurence_angles, levels=self._num_unique_angles)
        cooccurence_matrix = np.sum(cooccurence_matrix, axis=3)[:, :, 0]
        return mt.features.texture.haralick_features([cooccurence_matrix], return_mean=True)

    def _calculate_haralick_textures(self, dominant_angles):
        num_unique_angles = self._num_unique_angles
        dominant_angles_max, dominant_angles_min = dominant_angles.max(), dominant_angles.min()
        dominant_angles_binned = (dominant_angles - dominant_angles_min) / (dominant_angles_max - dominant_angles_min + np.finfo(float).eps) * (num_unique_angles - 1)
        dominant_angles_binned = np.round(dominant_angles_binned).astype(int)

        shape = dominant_angles_binned.shape
        haralick_image = np.empty(shape + (13,))
        haralick_image[:] = np.nan
        height, width, depth = shape

        for z in range(1, depth - 1) if self._is_3D else range(depth):
            for y, x in product(range(height), range(width)):
                if self._mask_array[y, x, z]:
                    haralick_image[y, x, z, :] = self._calculate_haralick_feature_values(dominant_angles_binned[:, :, z], x, y)
        return haralick_image

    def execute(self):
        svd_radius = self._svd_radius
        img_array = self._img_array
        
        cropped_min_x = max(self.mask_min_x - svd_radius, 0)
        cropped_min_y = max(self.mask_min_y - svd_radius, 0)
        cropped_min_z = max(self.mask_min_z - 1, 0)
        cropped_max_x = min(self.mask_max_x + svd_radius, img_array.shape[1])
        cropped_max_y = min(self.mask_max_y + svd_radius, img_array.shape[0])
        cropped_max_z = min(self.mask_max_z + 1, img_array.shape[2])

        extended_below = self.mask_min_z > 0
        extended_above = self.mask_max_z < img_array.shape[2]

        cropped_image = img_array[cropped_min_y:cropped_max_y, cropped_min_x:cropped_max_x, cropped_min_z:cropped_max_z]

        if cropped_image.max() > 1:
            cropped_image = cropped_image / cropped_image.max()

        dx = np.gradient(cropped_image, axis=1)
        dy = np.gradient(cropped_image, axis=0)
        dz = np.gradient(cropped_image, axis=2) if self._is_3D else np.zeros(dx.shape)

        if extended_below:
            dx, dy, dz = dx[:, :, 1:], dy[:, :, 1:], dz[:, :, 1:]
        if extended_above:
            dx, dy, dz = dx[:, :, :-1], dy[:, :, :-1], dz[:, :, :-1]

        dominant_angles = _svd_dominant_angles(dx, dy, dz, svd_radius)
        angles_shape = dominant_angles.shape

        haralick_features = np.empty(angles_shape[0:3] + (13, 2 if self._is_3D else 1,))
        for angle_index in range(angles_shape[3]):
            haralick_features[:, :, :, :, angle_index] = self._calculate_haralick_textures(dominant_angles[:, :, :, angle_index])

        collage_output = np.empty(img_array.shape + haralick_features.shape[3:5])
        collage_output[:] = np.nan

        collage_output[self.mask_min_y:self.mask_max_y, self.mask_min_x:self.mask_max_x, self.mask_min_z:self.mask_max_z, :, :] = haralick_features

        if not self._is_3D:
            collage_output = np.squeeze(collage_output, 4)
            collage_output = np.squeeze(collage_output, 2)

        self.collage_output = collage_output
        return collage_output


# ==========================================
# 2. PIPELINE CORE LOGIC
# ==========================================

def preprocess_to_isotropic_lps(sitk_image, is_mask=False, new_spacing=(1.0, 1.0, 1.0)):
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation('LPS')
    sitk_image = orient_filter.Execute(sitk_image)

    orig_spacing = sitk_image.GetSpacing()
    orig_size = sitk_image.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    resample.SetOutputPixelType(sitk.sitkUInt8 if is_mask else sitk.sitkFloat32)
    return resample.Execute(sitk_image)


def get_crop_coords(guide_mask, target_shape):
    if not np.any(guide_mask):
        raise ValueError("Guide mask is empty — cannot compute crop centre.")

    coords = np.argwhere(guide_mask)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    center_z = (z_min + z_max) // 2
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    orig_z, orig_y, orig_x = guide_mask.shape
    tz, ty, tx = target_shape

    def _clamp(center, size, orig):
        start = max(0, center - size // 2)
        end = start + size
        if end > orig:
            end = orig
            start = max(0, end - size)
        if start < 0:
            start = 0
            end = min(orig, size)
        return start, end

    z_start, z_end = _clamp(center_z, tz, orig_z)
    y_start, y_end = _clamp(center_y, ty, orig_y)
    x_start, x_end = _clamp(center_x, tx, orig_x)
    return z_start, z_end, y_start, y_end, x_start, x_end


def normalize_volume_channelwise(tensor_zyx_c, mask_3d):
    normalized = np.zeros_like(tensor_zyx_c, dtype=np.float32)
    for c in range(tensor_zyx_c.shape[-1]):
        ch = tensor_zyx_c[:, :, :, c]
        valid = mask_3d & np.isfinite(ch)
        tissue_vals = ch[valid]
        if len(tissue_vals) == 0 or tissue_vals.max() == tissue_vals.min():
            continue
        v_min = tissue_vals.min()
        v_max = tissue_vals.max()
        normalized[:, :, :, c] = np.where(
            valid,
            (ch - v_min) / (v_max - v_min + 1e-8),
            0.0
        )
    return normalized


# ==========================================
# 3. BATCH PROCESSING PIPELINE
# ==========================================

def process_case(ct_path, label_path, output_dir):
    case_id = os.path.basename(ct_path).replace('.nii.gz', '').replace('.nii', '')
    logger.info(f"--- Processing Case: {case_id} ---")

    # Load CT Image
    image = sitk.ReadImage(ct_path)
    image_resampled_sitk = preprocess_to_isotropic_lps(image, is_mask=False)

    # Load Labels
    label_img = nib.load(label_path)
    label_img_data = np.transpose(label_img.get_fdata(), (2, 1, 0))  # XYZ -> ZYX
    unique_labels = np.unique(label_img_data)
    is_pdac_case = (TUMOR_LABEL in unique_labels)

    # Resample Label relative to CT reference grid
    label_raw = sitk.ReadImage(label_path)
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation('LPS')
    label_lps = orient_filter.Execute(label_raw)

    resampler_labels = sitk.ResampleImageFilter()
    resampler_labels.SetReferenceImage(image_resampled_sitk)
    resampler_labels.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_labels.SetOutputPixelType(sitk.sitkUInt8)
    resampler_labels.SetDefaultPixelValue(0)
    labels_resampled_sitk = resampler_labels.Execute(label_lps)

    # Transform to Numpy arrays (ZYX)
    ct_resampled_numpy = sitk.GetArrayFromImage(image_resampled_sitk)
    resampled_labels_numpy = sitk.GetArrayFromImage(labels_resampled_sitk)

    # Build Masks
    tumor_3d_mask = (resampled_labels_numpy == TUMOR_LABEL)
    pancreas_3d_mask = (resampled_labels_numpy == PANCREAS_LABEL)
    combined_mask_labels_1_4_3d = tumor_3d_mask | pancreas_3d_mask

    # Define Crop Coordinates
    guide_mask = combined_mask_labels_1_4_3d.astype(bool) if is_pdac_case else pancreas_3d_mask.astype(bool)
    z_start, z_end, y_start, y_end, x_start, x_end = get_crop_coords(guide_mask, TARGET_SHAPE)

    # HU Clipping
    clipped_ct_numpy = np.clip(ct_resampled_numpy, HU_MIN, HU_MAX)

    # Crop Volumes
    _3d_cropped_ct_zyx = clipped_ct_numpy[z_start:z_end, y_start:y_end, x_start:x_end]
    _3d_cropped_mask_zyx = guide_mask[z_start:z_end, y_start:y_end, x_start:x_end]

    # Reshape for CoLIAGe inputs: ZYX -> YXZ
    _3d_image_for_collage_yxz = np.transpose(_3d_cropped_ct_zyx, (1, 2, 0))
    _3d_mask_for_collage_yxz = np.transpose(_3d_cropped_mask_zyx, (1, 2, 0))

    # Pad inputs for Boundary Safety
    SVD_RADIUS = 3
    PAD_Z = 1
    ct_padded = np.pad(_3d_image_for_collage_yxz,
                       ((PAD_Z, PAD_Z), (SVD_RADIUS, SVD_RADIUS), (SVD_RADIUS, SVD_RADIUS)),
                       mode='reflect')
    mask_padded = np.pad(_3d_mask_for_collage_yxz.astype(np.uint8),
                         ((PAD_Z, PAD_Z), (SVD_RADIUS, SVD_RADIUS), (SVD_RADIUS, SVD_RADIUS)),
                         mode='constant', constant_values=0)

    # Execute CoLIAGe
    collage_3d_instance = Collage(ct_padded, mask_padded, svd_radius=SVD_RADIUS, num_unique_angles=32)
    collage_3d_instance.execute()

    # Trim padding out
    raw = collage_3d_instance.collage_output
    raw_trimmed = raw[PAD_Z:-PAD_Z, SVD_RADIUS:-SVD_RADIUS, SVD_RADIUS:-SVD_RADIUS, :, :]

    # Transpose trimmed back to ZYX & flatten dimensions (Z, Y, X, 26)
    _3d_haralick_volume_zyx = np.transpose(raw_trimmed, (2, 0, 1, 3, 4))
    Z, Y, X = _3d_haralick_volume_zyx.shape[:3]
    _3d_haralick_26ch = _3d_haralick_volume_zyx.reshape(Z, Y, X, -1)

    # Build and Normalize 27-channel tensor
    ct_channel = _3d_cropped_ct_zyx[:, :, :, np.newaxis]
    tensor_raw = np.concatenate([ct_channel, _3d_haralick_26ch], axis=-1)
    mask_crop_bool = _3d_cropped_mask_zyx.astype(bool)
    tensor_normalized = normalize_volume_channelwise(tensor_raw, mask_crop_bool)

    # Append Case Label (Channel 28)
    label_channel = np.full(tensor_normalized.shape[:-1] + (1,), int(is_pdac_case), dtype=np.float32)
    tensor_with_label = np.concatenate([tensor_normalized, label_channel], axis=-1)

    # Define Output Subdirectories
    npy_dir = os.path.join(output_dir, "ALL_TENSORS", "npy_files")
    img_dir = os.path.join(output_dir, "ALL_METRICS", "ALL_IMAGES")
    mask_dir = os.path.join(output_dir, "ALL_METRICS", "ALL_MASKS")
    for d in [npy_dir, img_dir, mask_dir]:
        os.makedirs(d, exist_ok=True)

    # Save outputs
    tensor_out_path = os.path.join(npy_dir, f"{case_id}_tensor_128_27ch.npy")
    np.save(tensor_out_path, tensor_with_label)
    logger.info(f"Saved normalized 28ch tensor -> {tensor_out_path}")

    img_out_path = os.path.join(img_dir, f"{case_id}_image.npy")
    np.save(img_out_path, _3d_image_for_collage_yxz)

    mask_out_path = os.path.join(mask_dir, f"{case_id}_mask.npy")
    np.save(mask_out_path, _3d_mask_for_collage_yxz.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Process a batch of CT files for CoLIAGe feature generation.")
    parser.add_argument("--ct_dir", type=str, required=True, help="Path to directory containing input CT `.nii` or `.nii.gz` volumes.")
    parser.add_argument("--label_dir", type=str, required=True, help="Path to directory containing input label/mask files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output path for saving results.")
    args = parser.parse_args()

    # Find CT files
    #ct_patterns = [os.path.join(args.ct_dir, "*.nii"), os.path.join(args.ct_dir, "*.nii.gz")]
    #ct_files = []
    #for pattern in ct_patterns: ct_files.extend(glob.glob(pattern))
    
    # Find CT files (supports both a folder path or a direct file path)
    if os.path.isfile(args.ct_dir):
        ct_files = [args.ct_dir]
    else:
        ct_patterns = [os.path.join(args.ct_dir, "*.nii"), os.path.join(args.ct_dir, "*.nii.gz")]
        ct_files = []
        for pattern in ct_patterns:
            ct_files.extend(glob.glob(pattern))

    # Find Label files
    label_patterns = [os.path.join(args.label_dir, "**", "*.nii"), os.path.join(args.label_dir, "**", "*.nii.gz")]
    label_files = []
    for pattern in label_patterns:
        label_files.extend(glob.glob(pattern, recursive=True))

    logger.info(f"Found {len(ct_files)} CT files and {len(label_files)} label files.")

    # Generate lookup for label matches
    def get_case_id(path):
        # Extract the base filename without extensions (handles both .nii and .nii.gz)
        basename = os.path.basename(path)
        if basename.endswith('.nii.gz'):
            base_no_ext = basename[:-7]
        elif basename.endswith('.nii'):
            base_no_ext = basename[:-4]
        else:
            # Fallback split
            base_no_ext = basename.split('.')[0]
    
        # Split by underscore, drop the last component (e.g., "0000"), and rejoin
        parts = base_no_ext.split('_')
        if len(parts) > 1:
            return "_".join(parts[:-1])
        return base_no_ext
        #return os.path.basename(path).replace('.nii.gz', '').replace('.nii', '')

    label_lookup = {get_case_id(p): p for p in label_files}

    # Execute main processing loop
    processed_count = 0
    for ct_file in ct_files:
        case_id = get_case_id(ct_file)
        if case_id in label_lookup:
            try:
                process_case(ct_file, label_lookup[case_id], args.output_dir)
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process case {case_id}: {str(e)}", exc_info=True)
        else:
            logger.warning(f"No matching label found for CT volume: {case_id}. Skipping...")

    logger.info(f"Pipeline finished processing {processed_count}/{len(ct_files)} cases successfully.")

if __name__ == "__main__":
    main()