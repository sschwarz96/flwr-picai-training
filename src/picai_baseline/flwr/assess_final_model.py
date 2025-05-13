import json
import os
from pathlib import Path

import numpy as np
import torch
from picai_eval import Metrics
from picai_eval.eval import evaluate_case
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter

from src.picai_baseline.flwr.federated_training_methods import load_datasets
from src.picai_baseline.flwr.run_config import run_configuration
from src.picai_baseline.unet.training_setup.callbacks import fix_labels_shape
from src.picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run
from src.picai_baseline.unet.training_setup.loss_functions.focal import FocalLoss

WEIGHTS_FILE = Path("file")
FILE_NAME = "file_name.json"


def assess_model(model, loss_func, valid_gen, device):
    """Validate model per N epoch + export model weights"""

    # epoch, f = tracking_metrics['epoch'], tracking_metrics['fold_id']

    # for each validation sample
    lesion_results = []
    total_loss = 0.0  # Track loss
    step = 0
    model.eval()
    with torch.no_grad:
        for valid_data in valid_gen:

            try:
                valid_images = valid_data['data'].to(device)
                valid_labels = valid_data['seg']
                valid_labels_device = valid_data['seg'].to(device)
            except Exception:
                valid_images = torch.from_numpy(valid_data['data']).to(device)
                valid_labels_device = torch.from_numpy(valid_data['seg']).to(device)
                valid_labels = valid_data['seg']

            # bugfix for shape of targets
            valid_labels_device = fix_labels_shape(run_configuration, valid_labels_device)

            # Calculate validation loss
            outputs = model(valid_images)
            batch_loss = loss_func(outputs, valid_labels_device).item()
            total_loss += batch_loss
            step += 1

            # test-time augmentation
            valid_images_tta = [valid_images, torch.flip(valid_images, [4]).to(device)]

            # aggregate all validation predictions
            # gaussian blur to counteract checkerboard artifacts in
            # predictions from the use of transposed conv. in the U-Net
            preds = [torch.sigmoid(model(x))[:, 1, ...].detach().cpu().numpy() for x in valid_images_tta]

            # revert horizontally flipped tta image
            preds[1] = np.flip(preds[1], [3])

            # gaussian blur to counteract checkerboard artifacts in
            # predictions from the use of transposed conv. in the U-Net
            preds = np.mean([
                gaussian_filter(x, sigma=1.5)
                for x in preds
            ], axis=0)

            # extract lesion candidates
            preds = [
                extract_lesion_candidates(x)[0]
                for x in preds
            ]

            # evaluate detection maps of batch
            for y_det, y_true in zip(preds, valid_labels):
                y_list, *_ = evaluate_case(
                    y_det=y_det,
                    y_true=y_true.squeeze(),
                )

                # aggregate all validation evaluations
                lesion_results.append(y_list)

    avg_loss = total_loss / step
    # track validation metrics
    lesion_results = {idx: result for idx, result in enumerate(lesion_results)}
    valid_metrics = Metrics(lesion_results)

    num_pos = sum([y == 1 for y in valid_metrics.case_target.values()])
    num_neg = sum([y == 0 for y in valid_metrics.case_target.values()])

    print(f"Valid. Performance [Benign or Indolent PCa (n={num_neg}) \
        vs. csPCa (n={num_pos})]:\nRanking Score = {valid_metrics.score:.3f},\
        AP = {valid_metrics.AP:.3f}, AUROC = {valid_metrics.auroc:.3f}", flush=True)

    best_f2, best_threshold = valid_metrics.F2
    best_f2_threshold = {"best_threshold": best_threshold, "best_f2": best_f2}
    tracking_metrics = {
        "average_precision": valid_metrics.AP,
        "auroc": valid_metrics.auroc,
        "ranking": valid_metrics.score,
        "best_f2": best_f2_threshold,
        "loss": avg_loss  # Include loss
    }
    return tracking_metrics


def main():
    trainloader, valloader, class_weights = load_datasets(fold_id=4)

    device = torch.device("cuda:0")
    model = neural_network_for_run(args=run_configuration, device=device)
    checkpoint = torch.load(WEIGHTS_FILE)
    print(f"Loading weights from {WEIGHTS_FILE}")
    model.load_state_dict(checkpoint)
    model.to(device)
    loss_func = FocalLoss(alpha=class_weights[-1], gamma=run_configuration.focal_loss_gamma).to(device)

    metrics = assess_model(model, loss_func, trainloader, device)

    outputs_dir = Path('/home/zimon/flwr-picai-training/outputs')
    final_results_dir = outputs_dir / 'final_results'

    with open(final_results_dir / FILE_NAME, 'w') as file:
        file.write(json.dumps(metrics))


# Using the special variable
# __name__
if __name__ == "__main__":
    main()
