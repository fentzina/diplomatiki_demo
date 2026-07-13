import os
import sys
import glob
import gzip
import math
import shutil
import random
import logging
from itertools import product
from enum import Enum, IntEnum

import matplotlib
matplotlib.use('Agg')  # headless server: no display, so never try to open a GUI window

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
import scipy.ndimage as ndi
import mahotas as mt
from scipy import linalg
from skimage.feature import graycomatrix
from skimage.util.shape import view_as_windows
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Local filesystem layout (replaces Colab's google.colab.drive mount) ─────
# Everything lives under BASE_DIR now. Point it at wherever you want your
# data + outputs to live on this machine (a mounted data disk, NFS share, etc).
BASE_DIR = os.environ.get('COLLAGE_BASE_DIR', '/data/collage_pdac')

output_drive_dir = os.path.join(BASE_DIR, 'outputs')
plots_dir        = os.path.join(output_drive_dir, 'plots')
os.makedirs(output_drive_dir, exist_ok=True)
os.makedirs(plots_dir,        exist_ok=True)

TARGET_SHAPE   = (128, 128, 128)   # cubic crop
HU_MIN, HU_MAX = -100, 600        # soft-tissue / pancreas window


# ═══════════════════════════════════════════════════════════════════════════
# NEW CELL — Download & unzip the 4 Zenodo batches, build the case file list,
# and do all "run-once" setup that used to happen inside the per-case cells
# (cloning panorama_labels, cloning collageradiomics, building label lookup).
# ═══════════════════════════════════════════════════════════════════════════

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

# ---- 3. One-time clone of the labels repo (was re-cloned every case before) -
if not os.path.isdir('panorama_labels'):
    os.system('git clone https://github.com/DIAGNijmegen/panorama_labels.git')

def _case_id(path):
    b = os.path.basename(path)
    return b.replace('.nii.gz', '').replace('.nii', '')

label_nii_files = (
    glob.glob(os.path.join('panorama_labels', '**', '*.nii'), recursive=True) +
    glob.glob(os.path.join('panorama_labels', '**', '*.nii.gz'), recursive=True)
)
label_files_dict = {_case_id(p): p for p in label_nii_files}
print(f"Found {len(label_nii_files)} label files. Label lookup built with {len(label_files_dict)} entries.")

# ---- 4. One-time clone + installs for CoLIAGe (was re-run every case before) -
if not os.path.isdir('collageradiomics'):
    os.system('git clone https://github.com/radxtools/collageradiomics')
os.system('pip install -q pydicom')
import collageradiomics
import pydicom

# ---- 5. Make sure the output folders exist -----------------------------
os.makedirs(os.path.join(BASE_DIR, 'ALL_TENSORS', 'npy files'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'ALL_METRICS', 'ALL_IMAGES'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'ALL_METRICS', 'ALL_MASKS'), exist_ok=True)

# ---- 6. Turn off inline plt.show() for the batch run (leave True to debug
#         a single case interactively) — figures are still created & saved
#         if you add plt.savefig calls, they're just not rendered inline. ---
SHOW_PLOTS = False
if not SHOW_PLOTS:
    plt.show = lambda *args, **kwargs: None


# ═══════════════════════════════════════════════════════════════════════════
# EVERYTHING BELOW THIS LINE IS YOUR ORIGINAL PER-CASE PIPELINE, UNCHANGED —
# just wrapped in a function so it can be called once per case, in a loop.
# ═══════════════════════════════════════════════════════════════════════════

def process_one_case(nifti_file_path, selected_case_number):
    """# 4.  LOAD CT IMAGE  (SimpleITK — keeps spatial metadata)"""

    image = sitk.ReadImage(nifti_file_path)
    print(f"Image size    : {image.GetSize()}")
    print(f"Image spacing : {image.GetSpacing()}")
    print(f"Image origin  : {image.GetOrigin()}")
    print(f"Pixel type    : {image.GetPixelIDTypeAsString()}")

    # Quick sanity visualisation — raw image, random slice
    original_numpy_zyx = sitk.GetArrayFromImage(image)
    random_z = random.randint(0, original_numpy_zyx.shape[0] - 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(original_numpy_zyx[random_z], cmap='gray')
    plt.title(f'Raw CT slice Z={random_z}')
    plt.axis('off')
    plt.show()

    """# 5.  LOAD LABELS  (nibabel for quick inspection, SimpleITK for processing)"""
    # NOTE: panorama_labels was cloned ONCE, and label_nii_files / label_files_dict /
    # _case_id() were built ONCE, in the setup cell above. We just reuse them here
    # for every case instead of re-cloning + re-globbing 2240 times.

    ct_files_dict = {selected_case_number: nifti_file_path}
    print(f"CT files   : {len(ct_files_dict)}")
    print(f"Label files available: {len(label_files_dict)}")

    # Select label file for current case
    if selected_case_number in label_files_dict:
        sample_label_file = label_files_dict[selected_case_number]
        print(f"Label file: {sample_label_file}")
    elif label_nii_files:
        sample_label_file = label_nii_files[0]
        print(f"WARNING: No exact match — using fallback: {sample_label_file}")
    else:
        raise FileNotFoundError("No label files found. Check panorama_labels clone.")

    # nibabel: quick inspection only (not used downstream for processing)
    label_img      = nib.load(sample_label_file)
    label_img_data = np.transpose(label_img.get_fdata(), (2, 1, 0))  # XYZ → ZYX
    unique_labels  = np.unique(label_img_data)
    is_pdac_case   = (1 in unique_labels)
    print(f"Unique label values: {unique_labels}")
    print(f"Is PDAC case: {is_pdac_case}")

    """# 6.  PREPROCESSING FUNCTION  (orient LPS + resample to 1×1×1mm)
    All geometry operations happen at the SimpleITK level BEFORE any numpy conversion.  Labels use NearestNeighbor interpolation.
    """
    # @title
    def preprocess_to_isotropic_lps(sitk_image, is_mask=False,
                                     new_spacing=(1.0, 1.0, 1.0)):
        """
        1. Orient to LPS using DICOMOrientImageFilter.
        2. Resample to isotropic 1×1×1 mm spacing.

        Parameters
        ----------
        sitk_image  : SimpleITK.Image
        is_mask     : bool — True → NearestNeighbor + UInt8 output
                            False → Linear + Float32 output
        new_spacing : tuple of 3 floats, default (1.0, 1.0, 1.0)
        """
        # Step 1: LPS orientation
        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation('LPS')
        sitk_image = orient_filter.Execute(sitk_image)

        # Step 2: Isotropic resampling
        orig_spacing = sitk_image.GetSpacing()
        orig_size    = sitk_image.GetSize()
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
        resample.SetInterpolator(
            sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
        )
        resample.SetOutputPixelType(
            sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
        )
        return resample.Execute(sitk_image)

    # ── Resample CT
    print(f"Original CT spacing: {image.GetSpacing()}")
    image_resampled_sitk = preprocess_to_isotropic_lps(image, is_mask=False)
    print(f"Resampled CT spacing: {image_resampled_sitk.GetSpacing()}")

    # ── Resample labels — use CT as reference to guarantee identical grid
    label_raw = sitk.ReadImage(sample_label_file)

    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation('LPS')
    label_lps = orient_filter.Execute(label_raw)

    resampler_labels = sitk.ResampleImageFilter()
    resampler_labels.SetReferenceImage(image_resampled_sitk)   # ← grid from CT
    resampler_labels.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler_labels.SetOutputPixelType(sitk.sitkUInt8)
    resampler_labels.SetDefaultPixelValue(0)
    labels_resampled_sitk = resampler_labels.Execute(label_lps)

    # ── Convert to numpy (ZYX order)
    ct_resampled_numpy     = sitk.GetArrayFromImage(image_resampled_sitk)   # (Z, Y, X)
    resampled_labels_numpy = sitk.GetArrayFromImage(labels_resampled_sitk)  # (Z, Y, X)

    assert ct_resampled_numpy.shape == resampled_labels_numpy.shape, (
        f"Shape mismatch after resampling: CT {ct_resampled_numpy.shape} "
        f"vs labels {resampled_labels_numpy.shape}"
    )
    print(f"CT numpy shape    : {ct_resampled_numpy.shape}")
    print(f"Labels numpy shape: {resampled_labels_numpy.shape}")

    """# 7.  BUILD 3D SEGMENTATION MASKS  (always 3D — never overwrite with 2D)"""

    TUMOR_LABEL    = 1
    PANCREAS_LABEL = 4

    tumor_3d_mask    = (resampled_labels_numpy == TUMOR_LABEL)    # (Z, Y, X) bool
    pancreas_3d_mask = (resampled_labels_numpy == PANCREAS_LABEL) # (Z, Y, X) bool
    combined_mask_labels_1_4_3d = tumor_3d_mask | pancreas_3d_mask

    print(f"tumor_3d_mask shape              : {tumor_3d_mask.shape}")
    print(f"pancreas_3d_mask shape           : {pancreas_3d_mask.shape}")
    print(f"combined_mask_labels_1_4_3d shape: {combined_mask_labels_1_4_3d.shape}")
    print(f"Tumor voxels   : {int(np.sum(tumor_3d_mask))}")
    print(f"Pancreas voxels: {int(np.sum(pancreas_3d_mask))}")

    # ── Find representative Z-slice (max combined mask coverage) for visualisation
    pixels_per_slice = np.sum(combined_mask_labels_1_4_3d, axis=(1, 2))
    z_slice_idx      = int(np.argmax(pixels_per_slice))
    print(f"Representative Z slice (max mask coverage): {z_slice_idx}")

    # ── PDAC-specific stats (only when label 1 is present)
    if is_pdac_case:
        tumor_present_slices_indices = np.where(np.any(tumor_3d_mask, axis=(1, 2)))[0]
        print(f"Slices with tumor (label 1): {len(tumor_present_slices_indices)}")

        pdac_areas_mm2 = []
        for z in tumor_present_slices_indices:
            n_px = int(np.sum(tumor_3d_mask[z]))
            area = n_px * (image_resampled_sitk.GetSpacing()[0] *
                           image_resampled_sitk.GetSpacing()[1])
            pdac_areas_mm2.append({'slice': z, 'area_mm2': area})

        if pdac_areas_mm2:
            max_area_entry = max(pdac_areas_mm2, key=lambda e: e['area_mm2'])
            print(f"Max tumor area: {max_area_entry['area_mm2']:.1f} mm² at Z={max_area_entry['slice']}")

        # 3D volume
        voxel_vol_mm3         = float(np.prod(image_resampled_sitk.GetSpacing()))
        total_tumor_vol_mm3   = int(np.sum(tumor_3d_mask)) * voxel_vol_mm3
        print(f"Total tumor volume: {total_tumor_vol_mm3:.1f} mm³ "
              f"({total_tumor_vol_mm3/1000:.3f} cm³)")

    """# 8.  HU CLIPPING  (on the resampled CT numpy array)"""

    clipped_ct_numpy = np.clip(ct_resampled_numpy, HU_MIN, HU_MAX)
    print(f"HU clip [{HU_MIN}, {HU_MAX}]")
    print(f"  Before: min={ct_resampled_numpy.min():.0f}  max={ct_resampled_numpy.max():.0f}")
    print(f"  After : min={clipped_ct_numpy.min():.0f}  max={clipped_ct_numpy.max():.0f}")
    print(f"  Shape : {clipped_ct_numpy.shape}")

    # Sanity guard: CT and mask must have identical spatial shape from here on
    assert clipped_ct_numpy.shape == combined_mask_labels_1_4_3d.shape, (
        f"Shape mismatch: CT {clipped_ct_numpy.shape} "
        f"vs mask {combined_mask_labels_1_4_3d.shape}"
    )

    assert z_slice_idx < clipped_ct_numpy.shape[0], (
        f"z_slice_idx {z_slice_idx} out of bounds "
        f"(volume has {clipped_ct_numpy.shape[0]} slices)"
    )

    """# 9.  VISUALISATION — combined mask overlay on clipped CT"""

    colors_combined = [(0, 0, 0, 0), (0.5, 0.5, 1.0, 1.0)]
    cmap_combined   = ListedColormap(colors_combined)
    norm_combined   = BoundaryNorm([0, 0.5, 1], cmap_combined.N)

    colors_tumor = [(0, 0, 0, 0), (1.0, 0.0, 0.0, 1.0)]
    cmap_tumor   = ListedColormap(colors_tumor)
    norm_tumor   = BoundaryNorm([0, 0.5, 1], cmap_tumor.N)

    combined_mask_2d = combined_mask_labels_1_4_3d[z_slice_idx]
    tumor_2d_mask    = tumor_3d_mask[z_slice_idx]

    plt.figure(figsize=(8, 8))
    plt.imshow(clipped_ct_numpy[z_slice_idx], cmap='gray', origin='lower')
    plt.imshow(combined_mask_2d, cmap=cmap_combined, norm=norm_combined,
               alpha=0.5, origin='lower')
    if is_pdac_case:
        plt.imshow(tumor_2d_mask, cmap=cmap_tumor, norm=norm_tumor,
                   alpha=0.7, origin='lower')
    plt.title(f'Combined mask (labels 1+4) — Z={z_slice_idx}  |  clipped CT')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    """# 10. CROP COORDINATES  (single definition — no duplicates)
    FIX: **guide mask is the TRUE segmentation shape, not a bounding box**.
    * PDAC cases  → label 1 | label 4 (tumor always within crop)
    * Non-PDAC    → label 4 only
    """

    # @title
    def get_crop_coords(guide_mask, target_shape):
        """
        Returns (z_start, z_end, y_start, y_end, x_start, x_end) for a
        fixed-size crop centred on the bounding box of guide_mask.

        Parameters
        ----------
        guide_mask   : (Z, Y, X) boolean numpy array
        target_shape : (tz, ty, tx) desired output dimensions
        """
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
            end   = start + size
            if end > orig:
                end   = orig
                start = max(0, end - size)
            if start < 0:
                start = 0
                end   = min(orig, size)
            return start, end

        z_start, z_end = _clamp(center_z, tz, orig_z)
        y_start, y_end = _clamp(center_y, ty, orig_y)
        x_start, x_end = _clamp(center_x, tx, orig_x)
        return z_start, z_end, y_start, y_end, x_start, x_end

    # Select guide mask based on case type
    if is_pdac_case:
        guide_mask = combined_mask_labels_1_4_3d.astype(bool)
        print("PDAC case: guide mask = label 1 + label 4")
    else:
        guide_mask = pancreas_3d_mask.astype(bool)
        print("Non-PDAC case: guide mask = label 4 (pancreas) only")

    assert np.any(guide_mask), \
        f"Guide mask is empty for {selected_case_number} — check label availability."

    z_start, z_end, y_start, y_end, x_start, x_end = get_crop_coords(
        guide_mask, TARGET_SHAPE
    )
    print(f"Crop coords → Z:[{z_start}:{z_end}]  Y:[{y_start}:{y_end}]  X:[{x_start}:{x_end}]")
    assert (z_end - z_start) == TARGET_SHAPE[0], "Z crop size wrong"
    assert (y_end - y_start) == TARGET_SHAPE[1], "Y crop size wrong"
    assert (x_end - x_start) == TARGET_SHAPE[2], "X crop size wrong"

    # @title
    logger = logging.getLogger('collageradiomics')
    logger.info('Logging set up.')


    def _svd_dominant_angles(dx, dy, dz, svd_radius):
        """Calculate a new numpy image containing the dominant angles for each voxel.

            :param dx: 3D numpy array of the pixel gradient in the x directions
            :type dx: numpy.ndarray

            :returns: image array with the dominant angle calculated at each voxel
            :rtype: numpy.ndarray
        """

        is_3D = dx.shape[2] > 1

        # create rolling windows
        svd_diameter = svd_radius * 2 + 1

        # the first (commented out) version would actually take a patch of one slice above and below,
        # but by convention, the third dimension is simply used for the gradient calculation;
        # the actual collection of nearby gradient values to run through the SVD calculation is
        # still done on a given 2D slice
        #window_shape = (svd_diameter, svd_diameter) + ((3 if is_3D else 1),)
        window_shape = (svd_diameter, svd_diameter, 1)
        logger.info(f'Window patch shape for dominant angle calculation = {window_shape}')
        dx_windows = view_as_windows(dx, window_shape)
        dy_windows = view_as_windows(dy, window_shape)
        dz_windows = view_as_windows(dz, window_shape)

        angles_shape = dx_windows.shape[0:3]
        dominant_angles_array = np.zeros(angles_shape + (2 if is_3D else 1,), np.single)

        # loop through each voxel and use SVD to calculate the dominant angle for that rolling window
        # centered on that x,y,z coordinate
        center_x_range = range(angles_shape[1])
        center_y_range = range(angles_shape[0])
        center_z_range = range(angles_shape[2])
        for x, y, z in product(center_x_range, center_y_range, center_z_range):
            dominant_angles_array[y, x, z, :] = _svd_dominant_angle(x, y, z, dx_windows, dy_windows, dz_windows)

        return dominant_angles_array


    def _svd_dominant_angle(x, y, z, dx_windows, dy_windows, dz_windows):
        """Calculates the dominate angle at the coordinate within the windows.

            :param x: x value of coordinate
            :type x: int
            :param y: y value of coordinate
            :type y: int
            :param z: z value of coordinate
            :type z: int
            :param dx_windows: dx windows of x, y shape to run svd upon (shape = rows, cols, slices, row_radius, col_radius, slice_radius)
            :type dx_windows: numpy.ndarray
            :param dy_windows: dy windows of x, y shape to run svd upon
            :type dy_windows: numpy.ndarray
            :param dz_windows: dy windows of x, y shape to run svd upon
            :type dz_windows: numpy.ndarray

            :returns: dominant angle at x, y
            :rtype: float
        """

        # extract the patch of pixel gradient values for this specific voxel
        dx_patch = dx_windows[y, x, z]
        dy_patch = dy_windows[y, x, z]
        dz_patch = dz_windows[y, x, z]

        is_3D = dx_windows.shape[2] > 1

        # flatten all N gradient values in this patch into an Nxd matrix to pass into svd
        window_area = dx_patch.size
        flattened_gradients = np.zeros((window_area, (3 if is_3D else 2)))
        matrix_order = 'F' # fortran-style to be consistent with original matlab implementation
        flattened_gradients[:, 0] = np.reshape(dx_patch, window_area, order=matrix_order)
        flattened_gradients[:, 1] = np.reshape(dy_patch, window_area, order=matrix_order)
        if is_3D:
            flattened_gradients[:, 2] = np.reshape(dz_patch, window_area, order=matrix_order)

        # calculate svd
        _, _, v = linalg.svd(flattened_gradients)

        # extract results from the first column (in matlab this would be the first row)
        dominant_y = v[0,0]
        dominant_x = v[1,0]

        # calculate the dominant angle for this voxel
        dominant_angle = math.atan2(dominant_y, dominant_x)

        if is_3D:
            # also include the secondary angle
            dominant_z = v[2,0]
            secondary_angle = math.atan2(dominant_z, math.sqrt(dominant_x ** 2 + dominant_y ** 2))
            return (dominant_angle, secondary_angle)
        else:
            return dominant_angle

    class HaralickFeature(IntEnum):
        """Enumeration Helper For Haralick Features

            :param IntEnum: Enumeration Helper For Haralick Features
            :type IntEnum: HaralickFeature
        """
        AngularSecondMoment = 0
        Contrast = 1
        Correlation = 2
        SumOfSquareVariance = 3
        SumAverage = 4
        SumVariance = 5
        SumEntropy = 6
        Entropy = 7
        DifferenceVariance = 8
        DifferenceEntropy = 9
        InformationMeasureOfCorrelation1 = 10
        InformationMeasureOfCorrelation2 = 11
        MaximalCorrelationCoefficient = 12


    class DifferenceVarianceInterpretation(Enum):
        """ Feature 10 has two interpretations, as the variance of |x-y|
            or as the variance of P(|x-y|).
            See: https://ieeexplore.ieee.org/document/4309314

            :param Enum: Enumeration Helper For Haralick Features
            :type Enum: DifferenceVarianceInterpretation
        """
        XMinusYVariance = 0
        ProbabilityXMinusYVariance = 1


    class Collage:
        """This is the main object in the Collage calculation system. Usage: create a Collage object and then call the :py:meth:`execute` function.

            :param image_array: image to run collage upon
            :type image_array: numpy.ndarray
            :param mask_array: mask that correlates with the image
            :type mask_array: numpy.ndarray
            :param svd_radius: radius of svd. Defaults to 5.
            :type svd_radius: int, optional
            :param verbose_logging: This parameter is now ignored. Please use the python logging module.
            :type verbose_logging: bool, optional
            :param cooccurence_angles: list of angles to use in the cooccurence matrix. Defaults to [x*numpy.pi/4 for x in range(8)]
            :type cooccurence_angles: list, optional
            :param difference_variance_interpretation: Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
            :type difference_variance_interpretation: DifferenceVarianceInterpretation, optional
            :param haralick_window_size: size of rolling window for texture calculations. Defaults to -1.
            :type haralick_window_size: int, optional
            :param num_unique_angles: number of bins to use for the texture calculation. Defaults to 64.
            :type num_unique_angles: int, optional
        """


        @property
        def img_array(self):
            """
            The original image.

            :getter: Returns the original image array.
            :setter: Sets the original image array.
            :type: np.ndarray
            """
            return self._img_array

        @property
        def mask_array(self):
            """
            Array passed into Collage.

            :getter: Returns the original mask array.
            :setter: Sets the original mask array.
            :type: np.ndarray
            """
            return self._mask_array

        @property
        def is_3D(self):
            """
            Whether we are using 3D collage calculations (True) or 2D (False)
            """
            return self._is_3D

        @property
        def svd_radius(self):
            """
            SVD radius is used to calculate the pixel radius
            for the dominant angle calculation.

            :getter: Returns the SVD radius.
            :setter: Sets the SVD radius.
            :type: int
            """
            return self._svd_radius

        @property
        def verbose_logging(self):
            """
            This parameter is now ignored. Please use the python logging module.

            :getter: Returns True if on.
            :setter: Turns verbose logging off or on.
            :type: bool
            """
            return self._verbose_logging

        @property
        def cooccurence_angles(self):
            """
            Iterable of angles that will be used in the cooccurence matrix.

            :getter: Returns the Iterable of cooccurence angles.
            :setter: Sets the angles to be used in the cooccurence matrix.
            :type: int
            """
            return self._cooccurence_angles

        @property
        def difference_variance_interpretation(self):
            """
            Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].
            Defaults to DifferenceVarianceInterpretation.XMinusYVariance.

            :getter: Returns requested variance interpretation.
            :setter: Sets requested variance interpretation.
            :type: DifferenceVarianceInterpretation
            """
            return self._difference_variance_interpretation

        @property
        def haralick_window_size(self):
            """
            Number of pixels around each pixel to calculate a haralick texture.

            :getter: Returns requested number of pixels.
            :setter: Sets requested number of pixels.
            :type: int
            """
            return self._haralick_window_size

        @property
        def num_unique_angles(self):
            """
            Number of bins to use for texture calculations. Defaults to 64.

            :getter: Returns requested number of unique angles to bin into.
            :type: int
            """
            return self._num_unique_angles

        @property
        def collage_output(self):
            """
            Array representing collage upon the mask within the full images.
            If the input was 2D, the output will be height×width×13 where "13" is the number of haralick textures.
            If the input was 3D, the output will be height×width×depth×13x2 where "2" is the primary angle (element 0) or the secondary angle (element 1)

            The output will have numpy.nan values everywhere outside the masked region.

            :getter: Returns array the same shape as the original image with collage in the mask region.
            :type: numpy.ndarray
            """
            return self._collage_output

        @collage_output.setter
        def collage_output(self, value):
            self._collage_output = value


        def get_single_feature_output(self, which_feature):
            """
            Output a single collage output feature.
            If this was a 3D calculation, the output will be of size height×width×depth×2
            where the "2" represents the collage calculation from the primary angle (0) or secondary angle (1).

              :param which_feature: Either an integer from 0 to 12 (inclusive) or a HaralickFeature enum value
              :type which_feature HaralickFeature
            """
            if self.is_3D:
                return self.collage_output[:,:,:,which_feature,:]
            else:
                return self.collage_output[:,:,which_feature]


        def __init__(self,
                     img_array,
                     mask_array,
                     svd_radius=5,
                     verbose_logging=False,
                     cooccurence_angles=[x * np.pi/4 for x in range(8)],
                     difference_variance_interpretation=DifferenceVarianceInterpretation.XMinusYVariance,
                     haralick_window_size=-1,
                     num_unique_angles=64,
                     ):
            """Designated initializer for Collage

                :param image_array: image to run collage upon
                :type image_array: numpy.ndarray
                :param mask_array: mask that correlates with the image
                :type mask_array: numpy.ndarray
                :param svd_radius: radius of svd. Defaults to 5.
                :type svd_radius: int, optional
                :param verbose_logging: This parameter is now ignored. Please use the python logging module.
                :type verbose_logging: bool, optional
                :param cooccurence_angles: list of angles to use in the cooccurence matrix. Defaults to [x * np.pi/4 for x in range(8)]
                :type cooccurence_angles: list, optional
                :param difference_variance_interpretation: Feature 10 has two interpretations, as the variance of |x-y| or as the variance of P(|x-y|).].Defaults to DifferenceVarianceInterpretation.XMinusYVariance.
                :type difference_variance_interpretation: DifferenceVarianceInterpretation, optional
                :param haralick_window_size: size of rolling window for texture calculations. Defaults to -1.
                :type haralick_window_size: int, optional
                :param num_unique_angles: number of bins to use for the texture calculation. Defaults to 64.
                :type num_unique_angles: int, optional
            """

            logger.debug('Collage Module Initialized')

            # error checking
            if haralick_window_size == -1:
                self._haralick_window_size = svd_radius * 2 + 1
            else:
                self._haralick_window_size = haralick_window_size

            if self._haralick_window_size < 1:
                raise Exception('Haralick windows size must be at least 1 pixel.')

            if svd_radius < 1:
                raise Exception('SVD radius must be at least 1 pixel')

            if num_unique_angles < 1:
                raise Exception('num_unique_angles must contain at least 1 bin')

            if img_array.ndim < 2 or img_array.ndim > 3:
                raise Exception('Expected a 2D or 3D image.')

            if mask_array.shape != img_array.shape:
                raise Exception('Mask must be the same shape as image.')

            # Our minimum size for x & y is 50x50
            min_x_y_size = 50

            if img_array.shape[0] < min_x_y_size or img_array.shape[1] < min_x_y_size:
                raise Exception(f'Image size ({img_array.shape[0]}x{img_array.shape[1]}) unsupported. Image must be a minimum of a 50x50 for collage to run.')

            self._is_3D = img_array.ndim == 3
            logger.debug(f'Running 3D Collage = {self.is_3D}')

            self._img_array = img_array
            if not self.is_3D:
                # in the case of a single 2D slice, give it a third dimension of unit length
                self._img_array = self._img_array.reshape(self._img_array.shape + (1,))

            min_3D_slices = 3
            if self._img_array.shape[0] <  self._haralick_window_size or self._img_array.shape[1] < self._haralick_window_size or (self._is_3D and self._img_array.shape[2] < min_3D_slices):
                raise Exception(
                    f'Image is too small for a window size of {self._haralick_window_size} pixels.')

            # threshold mask
            uniqueValues = np.unique(mask_array)
            numberOfValues = len(uniqueValues)
            if numberOfValues > 2:
                logger.info(f'Warning: Mask is not binary. Considering all {numberOfValues} nonzero values in the mask as a value of True.')
            thresholded_mask_array = (mask_array != 0)

            # make correct shape
            thresholded_mask_array = thresholded_mask_array.reshape(self.img_array.shape)

            # extract rectangular area of mask
            non_zero_indices = np.argwhere(thresholded_mask_array)
            min_mask_coordinates = non_zero_indices.min(0)
            max_mask_coordinates = non_zero_indices.max(0)+1
            self.mask_min_x = min_mask_coordinates[1]
            self.mask_min_y = min_mask_coordinates[0]
            self.mask_min_z = min_mask_coordinates[2]
            self.mask_max_x = max_mask_coordinates[1]
            self.mask_max_y = max_mask_coordinates[0]
            self.mask_max_z = max_mask_coordinates[2]

            cropped_mask_array = thresholded_mask_array[self.mask_min_y:self.mask_max_y,
                                                        self.mask_min_x:self.mask_max_x,
                                                        self.mask_min_z:self.mask_max_z]

            # store variables internally
            self._mask_array = cropped_mask_array

            self._svd_radius = svd_radius
            self._verbose_logging = verbose_logging

            self._cooccurence_angles = cooccurence_angles
            self._difference_variance_interpretation = difference_variance_interpretation

            self._num_unique_angles = num_unique_angles


        def _calculate_haralick_feature_values(self, img_array, center_x, center_y):

            """Gets the haralick texture feature values at the x, y, z coordinate.
    , pos[1]
                :param image_array: image to calculate texture
                :type image_array: numpy.ndarray
                :param center_x: x center of coordinate
                :type center_x: int
                :param center_y: y center of coordinate
                :type center_y: int
                :param window_size: size of window to pull for calculation
                :type window_size: int
                :param num_unique_angles: number of bins
                :type num_unique_angles: int
                :param haralick_feature: desired haralick feature
                :type haralick_feature: HaralickFeature

                :returns: A 13x1 vector of haralick texture at the coordinate.
                :rtype: numpy.ndarray
            """
            # extract subpart of image (todo: pass in result from view_as_windows)
            window_size = self.haralick_window_size
            min_x = int(max(0, center_x - window_size / 2 - 1))
            min_y = int(max(0, center_y - window_size / 2 - 1))
            max_x = int(min(img_array.shape[1] - 1, center_x + window_size / 2 + 1))
            max_y = int(min(img_array.shape[0] - 1, center_y + window_size / 2 + 1))
            cropped_img_array = img_array[min_y:max_y, min_x:max_x]

            # co-occurence matrix of all 8 directions and sum them
            cooccurence_matrix = graycomatrix(cropped_img_array, [1], self.cooccurence_angles, levels=self.num_unique_angles)
            cooccurence_matrix = np.sum(cooccurence_matrix, axis=3)
            cooccurence_matrix = cooccurence_matrix[:, :, 0]

            # extract haralick using mahotas library
            return mt.features.texture.haralick_features([cooccurence_matrix], return_mean=True)


        def _calculate_haralick_textures(self, dominant_angles):
            """Gets haralick texture values

                :param dominant_angles_array: An image of the dominant angles at each voxel
                :type dominant_angles_[:,:,:,feature_index]array: numpy.ndarray
                :param desired_haralick_feature: which feature to calculate
                :type desired_haralick_feature: Haralick Feature
                :param num_unique_angles: number of bins
                :type num_unique_angles: int
                :param haralick_window_size: size of window around pixels to calculate haralick value

                :returns: An hxwxdx13 set of haralick texture.
                :rtype: numpy.ndarray
            """

            # rescale from 0 to (num_unique_angles-1)
            num_unique_angles = self.num_unique_angles
            logger.debug(f'Rescaling dominant angles to {num_unique_angles} unique values.')
            dominant_angles_max = dominant_angles.max()
            dominant_angles_min = dominant_angles.min()
            dominant_angles_binned = (dominant_angles - dominant_angles_min) / (dominant_angles_max - dominant_angles_min + np.finfo(float).eps) * (num_unique_angles - 1)
            dominant_angles_binned = np.round(dominant_angles_binned).astype(int)
            logger.debug(f'Rescaling dominant angles done.')

            # prepare output
            shape = dominant_angles_binned.shape
            haralick_image = np.empty(shape + (13,))
            haralick_image[:] = np.nan

            # the haralick is calculated for each slice separately
            height, width, depth = shape

            logger.debug(f'dominant_angles_binned shape is {shape} mask shape is {self.mask_array.shape}')

            # In 3D, we extended the dominant angles by one slice in each direction, so now we need to trim those off.
            for z in range(1, depth - 1) if self.is_3D else range(depth):
                for y,x in product(range(height), range(width)):
                    if self.mask_array[y,x,z]:
                        haralick_image[y,x,z,:] = self._calculate_haralick_feature_values(dominant_angles_binned[:,:,z], x, y)

            return haralick_image


        def execute(self):
            """Begins haralick calculation.

                :returns: An image at original size that only has the masked section filled in with collage calculations.
                :rtype: numpy.ndarray
            """

            svd_radius = self.svd_radius

            # mask location
            mask_min_x = int(self.mask_min_x)
            mask_min_y = int(self.mask_min_y)
            mask_min_z = int(self.mask_min_z)
            mask_max_x = int(self.mask_max_x)
            mask_max_y = int(self.mask_max_y)
            mask_max_z = int(self.mask_max_z)

            mask_width  = mask_max_x - mask_min_x
            mask_height = mask_max_y - mask_min_y
            mask_depth  = mask_max_z - mask_min_z

            img_array = self.img_array

            # extend the mask outwards a bit (up to the edge of the image) to handle the svd radius
            cropped_min_x = max(mask_min_x - svd_radius, 0)
            cropped_min_y = max(mask_min_y - svd_radius, 0)
            cropped_min_z = max(mask_min_z - 1         , 0) # for 3D, we just extend 1 slice in both directions
            cropped_max_x = min(mask_max_x + svd_radius, img_array.shape[1])
            cropped_max_y = min(mask_max_y + svd_radius, img_array.shape[0])
            cropped_max_z = min(mask_max_z + 1         , img_array.shape[2])

            extended_below = mask_min_z > 0
            extended_above = mask_max_z < img_array.shape[2]

            cropped_image = img_array[cropped_min_y:cropped_max_y,
                                      cropped_min_x:cropped_max_x,
                                      cropped_min_z:cropped_max_z]

            logger.debug(f'Image shape = {img_array.shape}')
            logger.debug(f'Mask size = {mask_height}x{mask_width}x{mask_depth}')
            logger.debug(f'Image shape (cropped and padded) = {cropped_image.shape}')

            # ensure the image values range from 0-1
            if cropped_image.max() > 1:
                logger.debug(f'Note: Dividing image values by {cropped_image.max()} to convert to 0-1 range')
                cropped_image = cropped_image / cropped_image.max()

            # calculate x,y,z gradients
            logger.debug('Calculating pixel gradients:')
            dx = np.gradient(cropped_image, axis=1)
            dy = np.gradient(cropped_image, axis=0)
            dz = np.gradient(cropped_image, axis=2) if self.is_3D else np.zeros(dx.shape)

            if extended_below:
                dx = dx[:,:,1:]
                dy = dy[:,:,1:]
                dz = dz[:,:,1:]

            if extended_above:
                dx = dx[:,:,:-1]
                dy = dy[:,:,:-1]
                dz = dz[:,:,:-1]

            self.dx = dx
            self.dy = dy
            self.dz = dz
            logger.debug('Calculating pixel gradients done.')

            # calculate dominant angles of each patch
            logger.debug(f'Calculating dominant gradient angles using SVD for each image patch of size {svd_radius}x{svd_radius}')
            dominant_angles = _svd_dominant_angles(dx, dy, dz, svd_radius)
            self.dominant_angles = dominant_angles
            angles_shape = dominant_angles.shape
            logger.debug('Calculating dominant gradient angles done.')
            logger.debug(f'Dominant angles shape = {angles_shape}')

            # calculate haralick features of the dominant angles
            logger.debug('Calculating haralick features of angles:')
            haralick_features = np.empty(angles_shape[0:3] + (13, 2 if self.is_3D else 1,))
            for angle_index in range(angles_shape[3]):
                logger.info(f'Calculating features for angle {angle_index}:')
                haralick_features[:,:,:,:,angle_index] = self._calculate_haralick_textures(dominant_angles[:,:,:,angle_index])
                logger.info(f'Calculating features for angle {angle_index} done.')
            logger.debug('Calculating haralick features of angles done.')

            # prepare an output full of "NaN's"
            collage_output = np.empty(img_array.shape + haralick_features.shape[3:5])
            collage_output[:] = np.nan

            # if a mask covers the whole image, we'll offset the edges as nans
            if mask_height == img_array.shape[0] and mask_width == img_array.shape[1]:
                y_offset = int((img_array.shape[0] - dominant_angles.shape[0]) / 2)
                mask_min_y += y_offset
                mask_max_y -= y_offset
                x_offset = int((img_array.shape[1] - dominant_angles.shape[1]) / 2)
                mask_min_x += x_offset
                mask_max_x -= x_offset

            # insert the haralick output into the correct spot
            collage_output[mask_min_y:mask_max_y,
                           mask_min_x:mask_max_x,
                           mask_min_z:mask_max_z,
                           :, :] = haralick_features

            # remove the singleton third dimension from the output
            if not self.is_3D:
                collage_output = np.squeeze(collage_output, 4)
                collage_output = np.squeeze(collage_output, 2)

            # output
            self.collage_output = collage_output
            logger.debug(f'Output shape = {collage_output.shape}')
            return collage_output
    # Cell modified to force re-execution and define the Collage class.
    # Re-added comment to trigger execution again after fixing the IntEnum import issue.

    """# 12. CoLIAGe INPUT PREPARATION
    FIX: **_3d_cropped_mask_zyx is the TRUE segmentation shape, NOT a rectangular bounding box.**
    """

    _3d_cropped_ct_zyx   = clipped_ct_numpy[z_start:z_end, y_start:y_end, x_start:x_end]
    _3d_cropped_mask_zyx = guide_mask[z_start:z_end, y_start:y_end, x_start:x_end]

    print(f"Cropped CT   (ZYX): {_3d_cropped_ct_zyx.shape}")
    print(f"Cropped mask (ZYX): {_3d_cropped_mask_zyx.shape}")

    assert _3d_cropped_ct_zyx.shape == _3d_cropped_mask_zyx.shape, \
        "CT / mask shape mismatch after crop"
    assert np.any(_3d_cropped_mask_zyx), \
        "Cropped mask is empty — pancreas not within crop window."

    active_voxels = int(np.sum(_3d_cropped_mask_zyx))
    print(f"Active mask voxels: {active_voxels} / {_3d_cropped_mask_zyx.size} "
          f"({100*active_voxels/_3d_cropped_mask_zyx.size:.1f}%)")

    # CoLIAGe expects (Height, Width, Depth) = (Y, X, Z) — transpose ZYX → YXZ
    _3d_image_for_collage_yxz = np.transpose(_3d_cropped_ct_zyx,   (1, 2, 0))
    _3d_mask_for_collage_yxz  = np.transpose(_3d_cropped_mask_zyx, (1, 2, 0))

    print(f"CoLIAGe input CT   (YXZ): {_3d_image_for_collage_yxz.shape}")
    print(f"CoLIAGe input mask (YXZ): {_3d_mask_for_collage_yxz.shape}")
    assert _3d_image_for_collage_yxz.shape[2] >= 3, \
        f"Depth {_3d_image_for_collage_yxz.shape[2]} < 3 — not enough for 3D CoLIAGe"

    """## Quick visualisation: CT + mask overlay on representative slice"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ct_slice_vis   = clipped_ct_numpy[z_slice_idx]
    mask_slice_vis = guide_mask[z_slice_idx]
    axes[0].imshow(ct_slice_vis, cmap='gray', origin='lower')
    axes[0].imshow(mask_slice_vis, cmap=cmap_combined, norm=norm_combined,
                   alpha=0.5, origin='lower')
    axes[0].set_title(f'Full volume — Z={z_slice_idx}')
    axes[0].axis('off')

    # Show crop boundary on the same slice (if this slice falls in crop)
    if z_start <= z_slice_idx < z_end:
        crop_ct   = _3d_cropped_ct_zyx[z_slice_idx - z_start]
        crop_mask = _3d_cropped_mask_zyx[z_slice_idx - z_start]
        axes[1].imshow(crop_ct, cmap='gray', origin='lower')
        axes[1].imshow(crop_mask, cmap=cmap_combined, norm=norm_combined,
                       alpha=0.5, origin='lower')
        axes[1].set_title(f'Cropped 128³ region — Z={z_slice_idx}')
    else:
        axes[1].set_title('z_slice_idx outside crop range')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    """## IMPORTS FOR COLIAGE"""
    # NOTE: the collageradiomics repo clone, pydicom install, and imports were
    # already done ONCE in the setup cell above.

    # @title
    # Helper functions for visualization

    def show_colored_image(figure, axis, image_data, colormap=plt.cm.jet):
        """Helper method to show a colored image in matplotlib.


            :param figure: figure upon which to display
            :type figure: matplotlib.figure.Figure
            :param axis: axis upon which to display
            :type axis: matplotlib.axes.Axes
            :param image_data: image to display
            :type image_data: numpy.ndarray
            :param colormap: color map to convert for display. Defaults to plt.cm.jet.
            :type colormap: matplotlib.colors.Colormap, optional
        """

        if image_data.ndim == 3:
            image_data = image_data[:,:,0]
        image = axis.imshow(image_data, cmap=colormap)
        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)
        figure.colorbar(image, cax=colorbar_axis)


    def create_highlighted_rectangle(x, y, w, h):
        """Creates a matplotlib Rectangle object for a highlight effect


            :param x: x location to start rectangle
            :type x: int
            :param y: y location to start rectangle
            :type y: int
            :param w: width of rectangle
            :type w: int
            :param h: height of rectangle
            :type h: int

            :returns: Rectangle used to highlight within a plot
            :rtype: matplotlib.patches.Rectangle
        """
        return Rectangle((x, y), w, h, linewidth=3, edgecolor='cyan', facecolor='none')


    def highlight_rectangle_on_image(image_data, min_x, min_y, w, h, colormap=plt.cm.gray):
        """Highlights a rectangle on an image at the passed in coordinate.


            :param image_data: image to highlight
            :type image_data: numpy.ndarray
            :param min_x: x location to start highlight
            :type min_x: int
            :param min_y: y location to start highlight
            :type min_y: int
            :param w: width of highlight rectangle
            :type w: int
            :param h: height of highlight rectangle
            :type h: int
            :param colormap: color map to convert for display. Defaults to plt.cm.jet.
            :type colormap: matplotlib.colors.Colormap, optional

            :returns: image array with highlighted rectangle
            :rtype: numpy.ndarray
        """
        figure, axes = plt.subplots(1, 2, figsize=(15, 15))

        # Highlight window within image.
        show_colored_image(figure, axes[0], image_data, colormap)
        axes[0].add_patch(create_highlighted_rectangle(min_x, min_y, w, h))

        # Crop window.
        cropped_array = image_data[min_y:min_y + h, min_x:min_x + w]
        axes[1].set_title(f'Cropped Region ({w}x{h})')
        show_colored_image(figure, axes[1], cropped_array, colormap)

        plt.show()

        return cropped_array

    """# 13. EXECUTE TRUE 3D CoLIAGe  — keep BOTH dominant angles"""

    # @title
    mask = _3d_mask_for_collage_yxz.astype(np.uint8)

    # Find bounding box CoLIAGe will compute
    ys, xs, zs = np.where(mask > 0)

    print(f"Mask shape          : {mask.shape}")
    print(f"Y extent: {ys.min()} → {ys.max()}  size={ys.max()-ys.min()}")
    print(f"X extent: {xs.min()} → {xs.max()}  size={xs.max()-xs.min()}")
    print(f"Z extent: {zs.min()} → {zs.max()}  size={zs.max()-zs.min()}")
    print(f"Total ROI voxels    : {mask.sum()}")

    # The exact slice CoLIAGe uses internally
    mask_min_y, mask_max_y = ys.min(), ys.max()
    mask_min_x, mask_max_x = xs.min(), xs.max()
    mask_min_z, mask_max_z = zs.min(), zs.max()

    print(f"\nExpected output array shape (Y,X,Z,13,2):")
    print(f"  ({mask_max_y - mask_min_y}, "
          f"{mask_max_x - mask_min_x}, "
          f"{mask_max_z - mask_min_z}, 13, 2)")

    # @title
    # ── Cell 36: EXECUTE TRUE 3D CoLIAGe — FIXED ────────────────────────────────
    print("\n=== EXECUTING TRUE 3D CoLIAGe ===")

    SVD_RADIUS = 3
    PAD_Z      = 1   # CoLIAGe always extends Z by exactly 1 slice internally

    # ── FIX: pre-pad so CoLIAGe never hits the image boundary
    ct_padded   = np.pad(_3d_image_for_collage_yxz,
                         ((PAD_Z, PAD_Z), (SVD_RADIUS, SVD_RADIUS), (SVD_RADIUS, SVD_RADIUS)),
                         mode='reflect')

    mask_padded = np.pad(_3d_mask_for_collage_yxz.astype(np.uint8),
                         ((PAD_Z, PAD_Z), (SVD_RADIUS, SVD_RADIUS), (SVD_RADIUS, SVD_RADIUS)),
                         mode='constant', constant_values=0)

    print(f"Padded CT shape   (YXZ): {ct_padded.shape}")
    print(f"Padded mask shape (YXZ): {mask_padded.shape}")

    # ── Run CoLIAGe on the padded volume
    collage_3d_instance = Collage(
        ct_padded,
        mask_padded,
        svd_radius        = SVD_RADIUS,
        num_unique_angles = 32
    )
    collage_3d_instance.execute()
    print("CoLIAGe execution successful.")

    print(f"Raw output shape (Y, X, Z, 13, 2): {collage_3d_instance.collage_output.shape}")

    # ── Trim the padding back out  →  original (Y, X, Z) spatial size
    raw = collage_3d_instance.collage_output   # (Ypad, Xpad, Zpad, 13, 2)
    raw_trimmed = raw[PAD_Z:-PAD_Z,             # trims Y by 1 → gives 128 ✓
                      SVD_RADIUS:-SVD_RADIUS,   # trims X by 3 → gives 128 ✓
                      SVD_RADIUS:-SVD_RADIUS,   # trims Z by 3 → gives 128 ✓
                      :, :]

    print(f"Trimmed output (Y, X, Z, 13, 2): {raw_trimmed.shape}")

    # ── The rest is identical to your original cell 36 ───────────────────────────
    # Transpose back to ZYX
    _3d_haralick_volume_zyx = np.transpose(raw_trimmed, (2, 0, 1, 3, 4))

    # Flatten 13 features × 2 angles → 26 channels: (Z, Y, X, 26)
    Z, Y, X = _3d_haralick_volume_zyx.shape[:3]
    _3d_haralick_26ch = _3d_haralick_volume_zyx.reshape(Z, Y, X, -1)

    # Dominant angles and gradients stay as they were
    _3d_dominant_angles_zyx = np.transpose(
        collage_3d_instance.dominant_angles, (2, 0, 1, 3)
    )
    _3d_dx_zyx = np.transpose(collage_3d_instance.dx, (2, 0, 1))
    _3d_dy_zyx = np.transpose(collage_3d_instance.dy, (2, 0, 1))
    _3d_dz_zyx = np.transpose(collage_3d_instance.dz, (2, 0, 1))

    print(f"Haralick volume  (Z, Y, X, 13, 2): {_3d_haralick_volume_zyx.shape}")
    print(f"Haralick 26ch    (Z, Y, X, 26)   : {_3d_haralick_26ch.shape}")

    # @title
    # Dynamic assertions (no hardcoded shapes)
    assert _3d_haralick_volume_zyx.shape == (Z, Y, X, 13, 2), \
        f"Unexpected Haralick shape: {_3d_haralick_volume_zyx.shape}"
    assert _3d_haralick_26ch.shape == (Z, Y, X, 26), \
        f"Unexpected 26ch shape: {_3d_haralick_26ch.shape}"
    assert not np.all(np.isnan(_3d_haralick_volume_zyx)), \
        "Haralick output is entirely NaN — mask was empty or CoLIAGe failed"

    print(f"\nTotal CNN channels: 1 CT + 26 Haralick = 27")

    # Crop with extra margin so the mask never touches the crop boundary —
    # this guarantees Collage's internal svd_radius extension is never clamped.
    SVD_RADIUS = 3
    COLIAGE_MARGIN = SVD_RADIUS + 2   # small safety buffer beyond the strict minimum
    TARGET_SHAPE_COLIAGE = tuple(s + 2 * COLIAGE_MARGIN for s in TARGET_SHAPE)  # e.g. (138,138,138)

    z_start_c, z_end_c, y_start_c, y_end_c, x_start_c, x_end_c = get_crop_coords(
        guide_mask, TARGET_SHAPE_COLIAGE
    )

    _3d_cropped_ct_zyx_wide   = clipped_ct_numpy[z_start_c:z_end_c, y_start_c:y_end_c, x_start_c:x_end_c]
    _3d_cropped_mask_zyx_wide = guide_mask[z_start_c:z_end_c, y_start_c:y_end_c, x_start_c:x_end_c]

    print("Wide crop shape:", _3d_cropped_ct_zyx_wide.shape)

    # Confirm the mask no longer touches the wide crop's boundary
    ys, xs, zs = np.where(_3d_cropped_mask_zyx_wide)
    print(f"Wide-crop mask bbox  Y:[{ys.min()},{ys.max()}]  X:[{xs.min()},{xs.max()}]  Z:[{zs.min()},{zs.max()}]")
    print(f"Wide-crop shape      Y:{_3d_cropped_ct_zyx_wide.shape[0]}  X:{_3d_cropped_ct_zyx_wide.shape[1]}  Z:{_3d_cropped_ct_zyx_wide.shape[2]}")

    # @title
    print("Mask bbox in the 128^3 crop, YXZ space:")
    print(f"  Y: [{collage_3d_instance.mask_min_y}, {collage_3d_instance.mask_max_y}]  "
          f"(crop height = {_3d_image_for_collage_yxz.shape[0]})")
    print(f"  X: [{collage_3d_instance.mask_min_x}, {collage_3d_instance.mask_max_x}]  "
          f"(crop width  = {_3d_image_for_collage_yxz.shape[1]})")
    print(f"  Z: [{collage_3d_instance.mask_min_z}, {collage_3d_instance.mask_max_z}]  "
          f"(crop depth  = {_3d_image_for_collage_yxz.shape[2]})")

    print("\nDistance from mask edge to crop edge (need >= SVD_RADIUS=3 on X/Y to avoid clamping):")
    print(f"  Y: left={collage_3d_instance.mask_min_y}, right={_3d_image_for_collage_yxz.shape[0]-collage_3d_instance.mask_max_y}")
    print(f"  X: left={collage_3d_instance.mask_min_x}, right={_3d_image_for_collage_yxz.shape[1]-collage_3d_instance.mask_max_x}")
    print(f"  Z: left={collage_3d_instance.mask_min_z}, right={_3d_image_for_collage_yxz.shape[2]-collage_3d_instance.mask_max_z}")

    """# 15. NORMALIZATION  (per-channel min-max over masked voxels only)
    FIX: normalization mask is _3d_cropped_mask_zyx (the cropped segmentation shape), not the full-volume mask re-sliced.
    """

    # @title
    def normalize_volume_channelwise(tensor_zyx_c, mask_3d):
        """
        Min-max normalize each channel independently over masked, finite voxels.
        Background (outside mask) remains 0.0.

        Parameters
        ----------
        tensor_zyx_c : np.ndarray  (Z, Y, X, C)
        mask_3d      : np.ndarray  (Z, Y, X) boolean
        Returns
        -------
        normalized   : np.ndarray  (Z, Y, X, C) float32  in [0, 1]
        """
        normalized = np.zeros_like(tensor_zyx_c, dtype=np.float32)
        for c in range(tensor_zyx_c.shape[-1]):
            ch          = tensor_zyx_c[:, :, :, c]
            valid       = mask_3d & np.isfinite(ch)
            tissue_vals = ch[valid]
            if len(tissue_vals) == 0 or tissue_vals.max() == tissue_vals.min():
                continue   # empty or constant channel → leave as zeros
            v_min = tissue_vals.min()
            v_max = tissue_vals.max()
            normalized[:, :, :, c] = np.where(
                valid,
                (ch - v_min) / (v_max - v_min + 1e-8),
                0.0
            )
        return normalized

    # Build 27-channel raw tensor: 1 CT + 26 Haralick
    ct_channel = _3d_cropped_ct_zyx[:, :, :, np.newaxis]           # (Z, Y, X, 1)
    tensor_raw  = np.concatenate([ct_channel, _3d_haralick_26ch], axis=-1)  # (Z, Y, X, 27)

    # Normalization mask = the cropped segmentation mask (true shape, not bbox)
    mask_crop_bool = _3d_cropped_mask_zyx.astype(bool)

    tensor_normalized = normalize_volume_channelwise(tensor_raw, mask_crop_bool)

    print(f"Final tensor shape : {tensor_normalized.shape}")   # (128, 128, 128, 27)
    print(f"Value range        : [{tensor_normalized.min():.4f}, {tensor_normalized.max():.4f}]")
    print(f"Dtype              : {tensor_normalized.dtype}")

    assert tensor_normalized.shape[-1] == 27, \
        f"Expected 27 channels, got {tensor_normalized.shape[-1]}"
    assert tensor_normalized.min() >= 0.0, "Normalization below 0"
    assert tensor_normalized.max() <= 1.0 + 1e-5, "Normalization above 1"

    """# 16. SAVE TENSOR  
    (FIX: save tensor_normalized, not final_3d_volume).

    Label (0: NON-PDAC, 1: PDAC) appended as the last channel so each .npy is self-contained.
    """

    label_channel    = np.full(
        tensor_normalized.shape[:-1] + (1,),
        int(is_pdac_case),
        dtype=np.float32
    )

    tensor_with_label = np.concatenate([tensor_normalized, label_channel], axis=-1)

    tensor_with_label.shape

    output_drive_dir = os.path.join(BASE_DIR, 'ALL_TENSORS', 'npy files')
    output_file_path = os.path.join(output_drive_dir, f"{selected_case_number}_tensor_128_27ch.npy")
    np.save(output_file_path, tensor_with_label)

    print(f"\nSaved tensor {tensor_with_label.shape} → {output_file_path}")
    print(f"  Channels 0    : CT (clipped, normalised)")
    print(f"  Channels 1-13 : Haralick × primary angle")
    print(f"  Channels 14-26: Haralick × secondary angle")
    print(f"  Channel  27   : PDAC label = {int(is_pdac_case)}")

    # Verify immediately after saving
    loaded = np.load(output_file_path)
    assert loaded.shape == (128, 128, 128, 28), f"Bad save: {output_file_path}"
    print(f"Verified: {output_file_path}")

    # ── ΣΩΣΤΗ ΑΠΟΘΗΚΕΥΣΗ ΕΙΚΟΝΑΣ (CT Image)
    output_path_local = os.path.join(BASE_DIR, 'ALL_METRICS', 'ALL_IMAGES', f"{selected_case_number}_image.npy")
    # Χρησιμοποιούμε το _3d_cropped_ct_zyx που περιέχει τις τιμές Hounsfield (HU)
    np.save(output_path_local, _3d_cropped_ct_zyx.astype(np.float32))
    print(f"Saved TRUE CT image tensor to local path: {output_path_local}")

    # ── ΣΩΣΤΗ ΑΠΟΘΗΚΕΥΣΗ ΜΑΣΚΑΣ (Mask / Label)
    output_path_local = os.path.join(BASE_DIR, 'ALL_METRICS', 'ALL_MASKS', f"{selected_case_number}_mask.npy")
    # Μετατροπή σε float32 για να ταιριάζει απόλυτα με το Colab σας
    np.save(output_path_local, _3d_cropped_mask_zyx.astype(np.float32))
    print(f"Saved TRUE mask tensor to local path: {output_path_local}")


    # Final shape: (128, 128, 128, 28)
    # Channels 0     : clipped CT (normalised)
    # Channels 1–13  : 13 Haralick features, primary dominant angle
    # Channels 14–26 : 13 Haralick features, secondary dominant angle
    # Channel  27    : PDAC label (0 or 1)

# ═══════════════════════════════════════════════════════════════════════════
# NEW CELL — run process_one_case() over all 2240 cases, with resume support
# (skips cases whose output .npy already exists) and per-case error handling
# (one bad case is logged and skipped, instead of killing the whole run).
# ═══════════════════════════════════════════════════════════════════════════

failed_cases   = []
processed_count = 0
TENSOR_OUTPUT_DIR = os.path.join(BASE_DIR, 'ALL_TENSORS', 'npy files')

for idx, nifti_file_path in enumerate(all_ct_files, start=1):
    selected_case_number = os.path.basename(nifti_file_path).replace('.nii.gz', '').replace('.nii', '')

    expected_output = os.path.join(TENSOR_OUTPUT_DIR, f"{selected_case_number}_tensor_128_27ch.npy")
    if os.path.exists(expected_output):
        print(f"[{idx}/{len(all_ct_files)}] {selected_case_number}: already processed — skipping.")
        continue

    print(f"\n{'='*70}\n[{idx}/{len(all_ct_files)}] Processing case: {selected_case_number}\n{'='*70}")
    try:
        process_one_case(nifti_file_path, selected_case_number)
        processed_count += 1
    except Exception as e:
        print(f"  !! FAILED: {selected_case_number} -> {e}")
        failed_cases.append((selected_case_number, str(e)))
    finally:
        plt.close('all')  # free all figures created during this case

print(f"\nDone. Successfully processed: {processed_count} / {len(all_ct_files)}")
print(f"Failed cases: {len(failed_cases)}")
if failed_cases:
    for cid, err in failed_cases:
        print(f"  - {cid}: {err}")
