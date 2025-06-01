#!/usr/bin/env python
"""
Run inference on the validation set, then compute and plot only the confusion matrix.
All units are metric (mm spacing, etc.)."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

# picai / baseline imports
from src.picai_baseline.flwr.federated_training_methods import load_datasets
from src.picai_baseline.unet.training_setup.callbacks import fix_labels_shape
from src.picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run
from src.picai_baseline.flwr.run_config import run_configuration
from report_guided_annotation import extract_lesion_candidates
from picai_eval import Metrics
from picai_eval.eval import evaluate_case

# Path to your trained weights and fold selection
WEIGHTS_FILE = Path(
    "/home/zimon/flwr-picai-training/outputs/final_results/DA/no_DP_DA_enabled/13-02-43/model_state_rank_0.45067658670752464_round_28.pth")
FOLD_ID = 4


def infer_and_collect(valloader, model, device):
    """
    Run inference with TTA and collect lesion_results for each case.
    """
    model.eval()
    lesion_results = []
    with torch.no_grad():
        for batch in valloader:
            # Move inputs to device
            imgs = torch.from_numpy(batch['data']).to(device) if not hasattr(batch['data'], 'to') else batch['data'].to(
                device)
            segs_dev = torch.from_numpy(batch['seg']).to(device) if not hasattr(batch['seg'], 'to') else batch[
                'seg'].to(device)
            segs = batch['seg']  # CPU labels

            # Fix shape and run model
            segs_dev = fix_labels_shape(segs_dev)
            tta_inputs = [imgs, torch.flip(imgs, dims=[4]).to(device)]
            preds = [
                torch.sigmoid(model(x))[:, 1, ...].cpu().numpy()
                for x in tta_inputs
            ]
            preds[1] = np.flip(preds[1], axis=3)
            preds = np.mean([gaussian_filter(p, sigma=1.5) for p in preds], axis=0)

            # Extract lesions and evaluate case-level maps
            for det_map, true_mask in zip(preds, segs):
                candidates = extract_lesion_candidates(det_map)[0]
                y_list, *_ = evaluate_case(y_det=candidates, y_true=true_mask.squeeze())
                lesion_results.append(y_list)

    return lesion_results


def main():
    # Load validation data
    _, valloader, class_weights = load_datasets(fold_id=FOLD_ID)

    # Prepare model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = neural_network_for_run(args=run_configuration, device=device)
    ckpt = torch.load(WEIGHTS_FILE, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)

    # Inference + collect results
    results = infer_and_collect(valloader, model, device)

    # Build Metrics object
    lesion_dict = {i: res for i, res in enumerate(results)}
    metrics = Metrics(lesion_dict)

    # Extract true labels and probabilities
    y_true = np.array(list(metrics.case_target.values()), dtype=int)
    y_prob = np.array(list(metrics.case_pred.values()), dtype=float)

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    ix = np.argmax(j_scores)
    best_thr_roc = thr[ix]
    print(f"Youden J max at thr={best_thr_roc:.3f}")
    y_pred = (y_prob >= best_thr_roc).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"ROC-optimal: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Print resultss
    print("Confusion matrix (counts):")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=3))

    order = [1, 0]

    # 2) Reorder both axes:
    cm_reordered = cm[order, :][:, order]
    # Now cm_reordered is [[TP, FN],
    #                      [FP, TN]]

    # 3) Define labels in that same order:
    labels = ['csPCa (1)', 'Benign/Indolent (0)']

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis('off')

    # create the table
    table = ax.table(
        cellText=cm_reordered,
        rowLabels=labels,
        colLabels=labels,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
