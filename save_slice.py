import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist, adjust_gamma

# Paths to your three modalities
modalities = {
    'T2W': '/home/zimon/flwr-picai-training/old_workdir/nnUNet_raw_data/Task2203_picai_baseline/imagesTr/10005_1000005_0002.nii.gz',
    'ADC': '/home/zimon/flwr-picai-training/old_workdir/nnUNet_raw_data/Task2203_picai_baseline/imagesTr/10005_1000005_0000.nii.gz',
    'HBV': '/home/zimon/flwr-picai-training/old_workdir/nnUNet_raw_data/Task2203_picai_baseline/imagesTr/10005_1000005_0001.nii.gz'
}

mask_path = '/home/zimon/flwr-picai-training/old_workdir/nnUNet_raw_data/Task2203_picai_baseline/labelsTr/10005_1000005.nii.gz'

# 1) define per-modality WL/WW and colormap
wl_ww = {
    'T2W': (100, 400),
    'ADC': (0,   2000),
    'HBV': (0,   800)     # tighten this until you see meaningful contrast
}
cmaps = {
    'T2W': 'bone',
    'ADC': 'bone',
    'HBV': 'magma'        # or 'hot', 'viridis', 'gray'
}

# load as before…
arrays   = {n: sitk.GetArrayFromImage(sitk.ReadImage(p)) for n,p in modalities.items()}
mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
idx      = int(np.argmax(mask_arr.sum(axis=(1,2))))
mask_slice = mask_arr[idx]

# 1) auto percentile‐based windowing
def auto_wl_norm(img, p_low=2, p_high=98):
    lo, hi = np.percentile(img, (p_low, p_high))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo)
    return adjust_gamma(img, 0.8)

# 2) manual WL/WW or grayscale fallback
wl_ww = {'T2W':(100,400), 'ADC':(0,2000)}
cmaps = {'T2W':'bone','ADC':'bone'}

# process all three modalities:
slices = {}
for name, arr in arrays.items():
    sl = arr[idx]
    if name == 'HBV':
        proc = auto_wl_norm(sl)         # auto-window + gamma
        cmap = 'gray'                   # simpler, avoids weird color artifacts
    else:
        wl, ww = wl_ww[name]
        proc = np.clip(sl, wl-ww/2, wl+ww/2)
        proc = (proc - proc.min())/(proc.max()-proc.min())
        proc = adjust_gamma(proc, 0.8)
        cmap = cmaps[name]
    slices[name] = (proc, cmap)

# 2×3 panel
fig, axs = plt.subplots(2,3,figsize=(12,8))
for col, name in enumerate(['T2W','ADC','HBV']):
    img, cmap = slices[name]
    axs[0,col].imshow(img, cmap=cmap)
    axs[0,col].set_title(f'{name} (raw)')
    axs[0,col].axis('off')
    axs[1,col].imshow(img, cmap=cmap)
    axs[1,col].imshow(mask_slice, cmap='Reds', alpha=0.3)
    axs[1,col].set_title(f'{name} + mask')
    axs[1,col].axis('off')
plt.tight_layout(pad=1)
plt.savefig("panel_fixed_hbv.png", dpi=300, bbox_inches="tight")
plt.show()