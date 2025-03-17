import SimpleITK as sitk

dicom_dir = r""
output_file = r""


def dicom_to_mha():
    """
    Convert a folder of DICOM files to a single MHA file.
    
    Parameters:
    - dicom_dir: Path to the directory containing DICOM files.
    - output_file: Path to save the output MHA file.
    """
    
    
    image = sitk.ReadImage(dicom_dir)
    
    # Save as .mha
    sitk.WriteImage(image, output_file)
    print(f"Saved: {output_file}")

# Example usage:
dicom_folder = "path/to/dicom/folder"  # Change this to your DICOM folder path
output_mha = "output.mha"  # Change this to your desired output file path

dicom_to_mha()
