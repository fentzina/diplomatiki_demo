import SimpleITK as sitk
import numpy as np
import sys

path = sys.argv[1]
img = sitk.ReadImage(path)
print("Size:", img.GetSize())
print("Spacing:", img.GetSpacing())
print("Origin:", img.GetOrigin())

arr = sitk.GetArrayFromImage(img)
tumor = (arr == 1)
pancreas = (arr == 4)

print(f"\nTumor voxels: {tumor.sum()}")
print(f"Pancreas voxels: {pancreas.sum()}")

if tumor.sum() > 0:
    tz, ty, tx = np.where(tumor)
    print(f"Tumor bbox (z,y,x): z[{tz.min()}-{tz.max()}] y[{ty.min()}-{ty.max()}] x[{tx.min()}-{tx.max()}]")
if pancreas.sum() > 0:
    pz, py, px = np.where(pancreas)
    print(f"Pancreas bbox (z,y,x): z[{pz.min()}-{pz.max()}] y[{py.min()}-{py.max()}] x[{px.min()}-{px.max()}]")
  
