import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import gzip
import shutil

# Define input and output filenames
gz_file = "./input/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b/10000_1000000.nii.gz"


# Load the file
img = nib.load(gz_file)
data = img.get_fdata()  # Convert to numpy array

# Show a middle slice
plt.imshow(data[:, :, data.shape[2]//2], cmap="gray")
plt.show()