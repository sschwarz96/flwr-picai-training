import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1) SETTINGS: adjust these paths and the slice index to suit your data
# ─────────────────────────────────────────────────────────────────────────────

# (a) Full paths to your files.
#     Replace these with wherever your .mha and .nii.gz actually live.
mha_path = '/home/zimon/flwr-picai-training/workdir/output/images/cspca-detection-map/cspca_detection_map_10005.mha'
nii_path = '/home/zimon/flwr-picai-training/input/picai_labels/csPCa_lesion_delineations/human_expert/original/10005_1000005.nii.gz'

patient = 10005

# (b) Choose the slice index you want to show (0-based).
#     E.g. slice_index = 50 will grab the 51st axial slice in each volume.
slice_index = 10

# (c) Where to save the side‐by‐side PNG.
#     You can overwrite this if you rerun multiple times.
output_png = f'/home/zimon/flwr-picai-training/metrics_res/comparison_slice_{patient}.png'

# ─────────────────────────────────────────────────────────────────────────────
# 2) LOAD VOLUMES (SimpleITK can read both .mha and NIfTI)
# ─────────────────────────────────────────────────────────────────────────────

# (a) Check that the files actually exist:
if not os.path.isfile(mha_path):
    raise FileNotFoundError(f"Cannot find MHA file: {mha_path}")
if not os.path.isfile(nii_path):
    raise FileNotFoundError(f"Cannot find NIfTI file: {nii_path}")

# (b) Read them in
vol_mha = sitk.ReadImage(mha_path)
vol_nii = sitk.ReadImage(nii_path)

# (c) Convert to NumPy arrays.
#     Note: SimpleITK’s GetArrayFromImage returns an array shaped [Z, Y, X].
arr_mha = sitk.GetArrayFromImage(vol_mha)  # shape = (num_slices, height, width)
arr_nii = sitk.GetArrayFromImage(vol_nii)  # same shape if orientations match

# ─────────────────────────────────────────────────────────────────────────────
# 3) EXTRACT ONE SLICE FROM EACH
# ─────────────────────────────────────────────────────────────────────────────

# (a) Verify that slice_index is within bounds
num_slices_mha = arr_mha.shape[0]
num_slices_nii = arr_nii.shape[0]
if slice_index < 0 or slice_index >= num_slices_mha:
    raise IndexError(f"slice_index = {slice_index} out of range for MHA volume (0–{num_slices_mha - 1})")
if slice_index < 0 or slice_index >= num_slices_nii:
    raise IndexError(f"slice_index = {slice_index} out of range for NIfTI volume (0–{num_slices_nii - 1})")

# (b) Grab the 2D slices
slice_mha = arr_mha[slice_index, :, :]
slice_nii = arr_nii[slice_index, :, :]

# ─────────────────────────────────────────────────────────────────────────────
# 4) PLOT SIDE‐BY‐SIDE AND SAVE AS PNG
# ─────────────────────────────────────────────────────────────────────────────

# (a) Create a figure with 1 row, 2 columns
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))

# (b) Show the MHA slice
ax0.imshow(slice_mha, cmap='gray')
ax0.set_title(f'Prediction Slice #{slice_index}. Max Prob 0.4')
ax0.axis('off')

# (c) Show the NIfTI (ground‐truth) slice
ax1.imshow(slice_nii, cmap='gray')
ax1.set_title(f'Ground Truth Slice #{slice_index}')
ax1.axis('off')

plt.tight_layout()

# (d) Save at high resolution (e.g. 300 DPI).
#     You can change bbox_inches='tight' if you want to trim margins.
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"→ Saved side‐by‐side comparison to: {output_png}")
