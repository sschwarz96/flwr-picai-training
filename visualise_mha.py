import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Load MHA file
file_path = "./input/images/10000/10000_1000000_adc.mha"
image = sitk.ReadImage(file_path)
array = sitk.GetArrayFromImage(image)  # Convert to numpy array

# Display middle slice
slice_idx = array.shape[0] // 2  # Choose middle slice
plt.imshow(array[slice_idx], cmap="gray")
plt.title(f"Slice {slice_idx}")
plt.axis("off")
plt.show()