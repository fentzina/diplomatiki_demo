import numpy as np
import os

FRACTION       = 0.4   # Μπορείτε να το αλλάξετε σε 0.5, 0.6 κλπ.
# Ορίστε τη διαδρομή για το αρχείο .txt με τα 462 απομονωμένα IDs
TXT_IDS_FILE   = "/home/student1/ftzina_thesis/output_pdac/filtered_pdac_cases_ids.txt" 

HETEROGEN_DIR  = os.path.expanduser(f'/home/student1/ftzina_thesis/output_pdac/intra_tumor_589_PDACs_{FRACTION}')
AGGREGATED_DIR = os.path.expanduser(f'/home/student1/ftzina_thesis/output_pdac/intra_tumor_589_PDACs_{FRACTION}')
txt_ids_path   = os.path.expanduser(TXT_IDS_FILE)

# 1. Φόρτωση των IDs από το αρχείο .txt
if not os.path.exists(txt_ids_path):
    raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο με τα IDs: {txt_ids_path}")

with open(txt_ids_path, 'r') as f:
    # Διαβάζει κάθε γραμμή, αφαιρεί τα κενά/newlines και αγνοεί τις άδειες γραμμές
    case_ids = [line.strip() for line in f if line.strip()]

print(f"Computing tumor volumes for {len(case_ids)} cases (fraction={FRACTION})...")

tumor_volumes = []
missing       = []

# 2. Υπολογισμός όγκων
for case_id in case_ids:
    c_path = os.path.join(HETEROGEN_DIR, f"{case_id}_central_mask.npy")
    p_path = os.path.join(HETEROGEN_DIR, f"{case_id}_peripheral_mask.npy")

    if not os.path.exists(c_path) or not os.path.exists(p_path):
        print(f"  WARNING: mask not found for {case_id} — using 0.")
        tumor_volumes.append(0.0)
        missing.append(case_id)
        continue

    c_mask = np.load(c_path).astype(bool)
    p_mask = np.load(p_path).astype(bool)
    vol    = int((c_mask | p_mask).sum())   # union = full tumor voxel count = volume in mm³
    tumor_volumes.append(float(vol))

tumor_volumes = np.array(tumor_volumes, dtype=np.float32)
valid         = tumor_volumes > 0

# 3. Εκτύπωση στατιστικών (με έλεγχο για να μην κρασάρει αν είναι άδειο)
print(f"\nTumor volumes (fraction={FRACTION}):")
print(f"  Cases with valid volume : {valid.sum()}")
print(f"  Cases with missing mask : {len(missing)}")

if valid.sum() > 0:
    print(f"  Mean volume             : {tumor_volumes[valid].mean():.0f} mm³  "
          f"= {tumor_volumes[valid].mean()/1000:.1f} cm³")
    print(f"  Range                   : [{tumor_volumes[valid].min():.0f}, "
          f"{tumor_volumes[valid].max():.0f}] mm³")
else:
    print("  Mean volume             : N/A")
    print("  Range                   : N/A")

# 4. Αποθήκευση αποτελεσμάτων και των IDs ως .npy για το επόμενο βήμα
out_path = os.path.join(AGGREGATED_DIR, "tumor_volumes.npy")
np.save(out_path, tumor_volumes)

# Προαιρετικά αποθηκεύουμε και τα IDs σε npy μορφή στον aggregated φάκελο αν χρειάζεται
np.save(os.path.join(AGGREGATED_DIR, f"case_ids_{FRACTION}.npy"), np.array(case_ids))

print(f"\nSaved volumes to: {out_path}")
