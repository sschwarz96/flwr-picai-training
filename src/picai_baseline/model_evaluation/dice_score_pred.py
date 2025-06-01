import SimpleITK as sitk
import numpy as np


def load_volume_as_array(path):
    """
    Reads a volume file (MetaImage .mha, NIfTI, etc.) and returns
    the voxel data as a NumPy array with shape (Z, Y, X).
    """
    img = sitk.ReadImage(path)  # auto-detects .mha/.nii/.nii.gz/etc.
    arr = sitk.GetArrayFromImage(img)  # returns a NumPy array
    return arr


def dice_score_binary(y_true, y_pred, epsilon=1e-6):
    """
    Computes the Dice coefficient between two binary (0/1 or bool) arrays.
    y_true, y_pred : NumPy arrays of identical shape with dtype bool or {0,1}.
    Returns Dice = (2 * |intersection| + ε) / (|y_true| + |y_pred| + ε)
    """
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    intersection = np.logical_and(y_true_bool, y_pred_bool).sum()
    return (2.0 * intersection + epsilon) / (y_true_bool.sum() + y_pred_bool.sum() + epsilon)


def compute_dice_from_volumes(
        mask_path,
        pred_path,
        mask_label=3,
        threshold=0.5
):
    """
    1. Loads the ground-truth label volume (mask_path) and the predicted
       probability volume (pred_path).
    2. Converts the label volume into a binary mask by selecting voxels == mask_label.
    3. Converts the probability volume into a binary mask by thresholding ≥ threshold.
    4. Computes and returns the Dice score.

    Arguments:
        mask_path   : str, filepath to the label/ground-truth volume (e.g., .mha or .nii).
        pred_path   : str, filepath to the predicted probability volume (e.g., .mha or .nii).
        mask_label  : int, the integer value in the ground-truth volume corresponding to “positive” (e.g., cancer=3).
                      All other values are treated as background. Default=3.
        threshold   : float, cutoff for converting probabilities to 0/1 mask. Default=0.5.

    Returns:
        dice : float, the Dice coefficient between the two binary masks.
    """
    # --- 1. Load volumes as NumPy arrays ---
    gt_array = load_volume_as_array(mask_path)
    pred_array = load_volume_as_array(pred_path)

    # --- 2. Ensure shapes match ---
    if gt_array.shape != pred_array.shape:
        raise ValueError(
            f"Shape mismatch: ground-truth volume shape {gt_array.shape} "
            f"!= prediction volume shape {pred_array.shape}"
        )

    # --- 3. Convert ground truth to binary where voxels == mask_label ---
    #     If gt_array has values {0, 1, 2, 3, ...}, and cancer is labeled as 3,
    #     then (gt_array == 3) yields a boolean mask where “True” = cancer.
    gt_binary = (gt_array == mask_label).astype(np.uint8)

    # --- 4. Convert prediction probabilities to binary by thresholding ≥ threshold ---
    #     If pred_array contains floating probabilities in [0,1], e.g. 0.4, 0.7, etc.,
    #     then (pred_array >= threshold) yields a boolean mask.
    pred_binary = (pred_array >= threshold).astype(np.uint8)

    # --- 5. Compute Dice between those two binary masks ---
    dice = dice_score_binary(gt_binary, pred_binary)
    return dice
# Example usage (replace with your file paths):
dice = compute_dice_from_volumes("/home/zimon/flwr-picai-training/input/picai_labels/csPCa_lesion_delineations/AI/Bosma22a/10397_1000403.nii.gz", "/home/zimon/flwr-picai-training/workdir/output/images/cspca-detection-map/cspca_detection_map_10397.mha", mask_label=1, threshold=0.4)
print(f"Dice score: {dice:.4f}")
