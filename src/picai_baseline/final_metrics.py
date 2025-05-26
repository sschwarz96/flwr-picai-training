#!/usr/bin/env python
"""
Evaluate the final csPCa U‑Net on a held‑out (test) fold and print / save
case‑level metrics (AP, AUROC, ranking‑score, best‑F2, etc.).

This script merges **pre‑/post‑processing & inference** from the Grand‑Challenge
`csPCaAlgorithm` wrapper **with** the **metric aggregation** logic used during
`validate_model`, so you obtain exactly comparable final numbers.

Example
-------
python evaluate_final_metrics.py \
    --weights   /path/to/best_model.pth \
    --test-dir  /data/test_fold/        \
    --out-json  /results/test_metrics.json

Notes
-----
* The test directory must contain one sub‑folder per case.
  Each case folder needs the three axial modalities              : *_t2w, *_adc, *_hbv
  and the reference segmentation                                 : *_seg.(nii|nii.gz|mha)
  Naming can be arbitrary apart from the modality substring.
* All computations run in **metric units** (millimetres for spacing).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# picai / baseline imports
from picai_prep.preprocessing import (
    Sample,
    PreprocessingSettings,
    crop_or_pad,
    resample_img,
)

from src.picai_baseline.flwr.federated_training_methods import load_datasets
from src.picai_baseline.unet.training_setup.callbacks import fix_labels_shape
from src.picai_baseline.unet.training_setup.preprocess_utils import z_score_norm
from src.picai_baseline.unet.training_setup.neural_network_selector import (
    neural_network_for_run,
)
from src.picai_baseline.unet.training_setup.loss_functions.focal import FocalLoss

from src.picai_baseline.flwr.run_config import RunConfig
from report_guided_annotation import extract_lesion_candidates

# evaluation helpers (same as in validate_model)
from picai_eval import Metrics
from picai_eval.eval import evaluate_case  # adjust import if paths differ

# --------------------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------------------
IMG_SPEC = {
    "image_shape": [16, 128, 128],  # [slices, rows, cols]
    "spacing": [3.0, 0.5, 0.5],  # [Δz, Δy, Δx] in mm
}
MODALITIES = ["_t2w", "_adc", "_hbv"]
MODALITY_ORDER = {mod: idx for idx, mod in enumerate(MODALITIES)}


def find_highest_rank_pth_paths(base_dir):
    highest_rank_paths = []

    # Loop over all experiment folders at the base level
    for exp_folder in os.listdir(base_dir):
        exp_path = os.path.join(base_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue

        highest_rank = -1
        highest_rank_file = None

        # Walk through all subdirectories
        for root, _, files in os.walk(exp_path):
            for fname in files:
                if fname.endswith('.pth'):
                    # Extract the rank number from filename
                    match = re.search(r'rank_([\d.]+)', fname)
                    if match:
                        rank = float(match.group(1))
                        if rank > highest_rank:
                            highest_rank = rank
                            highest_rank_file = os.path.join(root, fname)

        if highest_rank_file:
            highest_rank_paths.append(highest_rank_file)

    return highest_rank_paths


def sort_modalities(files: List[Path]) -> List[Path]:
    """Return files sorted so that t2w ⟶ adc ⟶ hbv."""

    def _index(p: Path):
        for m in MODALITIES:
            if m in p.name:
                return MODALITY_ORDER[m]
        return len(MODALITIES)

    return sorted(files, key=_index)


def evaluate_on_generator(model, test_gen, loss_func, args, device):
    """
    Run inference + case-level metrics (AP, AUROC, ranking, best-F2)
    on a test DataLoader. Works exactly like validate_model but
    with no weight updates and prints+saves only the metrics.
    """

    model.eval()
    lesion_results = []
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in test_gen:
            try:
                imgs = batch['data'].to(device)              # if already a Tensor
                segs_dev = batch['seg'].to(device)
                segs = batch['seg']                          # keep CPU version for eval
            except AttributeError:
                # came in as numpy arrays
                imgs = torch.from_numpy(batch['data']).to(device)
                segs_dev = torch.from_numpy(batch['seg']).to(device)
                segs = batch['seg']

            segs_dev = fix_labels_shape(segs_dev)

            # --- compute loss on raw outputs (optional) ---
            logits = model(imgs)
            batch_loss = loss_func(logits, segs_dev).item()
            total_loss += batch_loss
            steps += 1

            # --- test‐time augmentation & smoothing ---
            tta_inputs = [imgs, torch.flip(imgs, dims=[4]).to(device)]
            preds = [
                torch.sigmoid(model(x))[:, 1, ...].cpu().numpy()
                for x in tta_inputs
            ]
            preds[1] = np.flip(preds[1], axis=3)  # undo horizontal flip
            preds = np.mean([gaussian_filter(p, sigma=1.5) for p in preds], axis=0)

            # --- lesion candidate extraction & per‐case eval ---
            for det_map, true_mask in zip(preds, segs):
                candidates = extract_lesion_candidates(det_map)[0]
                y_list, *_ = evaluate_case(y_det=candidates, y_true=true_mask.squeeze())
                lesion_results.append(y_list)

    # --- aggregate metrics ---
    avg_loss = total_loss / max(1, steps)
    lesion_dict = {i: res for i, res in enumerate(lesion_results)}
    metrics = Metrics(lesion_dict)

    # print results
    n_pos = sum(v == 1 for v in metrics.case_target.values())
    n_neg = sum(v == 0 for v in metrics.case_target.values())
    print(f"Test performance [benign/indolent (n={n_neg}) vs. csPCa (n={n_pos})]:")
    print(f"  Ranking score = {metrics.score:.3f}")
    print(f"  Average precision = {metrics.AP:.3f}")
    print(f"  AUROC = {metrics.auroc:.3f}")

    best_f2, best_thr = metrics.F2
    print(f"  Best F2 = {best_f2:.3f} @ threshold {best_thr:.3f}")
    print(f"  (Mean loss = {avg_loss:.4f})")

    # return a dict exactly like validate_model
    return {
        "average_precision": metrics.AP,
        "auroc": metrics.auroc,
        "ranking": metrics.score,
        "best_f2": {"best_threshold": best_thr, "best_f2": best_f2},
        "loss": avg_loss,
    }


def main():
    weights = find_highest_rank_pth_paths(Path('/home/zimon/flwr-picai-training/outputs/final_results'))
    folders = os.listdir('/home/zimon/flwr-picai-training/outputs/final_results')

    with open(
            '/home/zimon/flwr-picai-training/workdir/results/UNet/overviews/Task2203_picai_baseline/PI-CAI_val-fold-4.json',
            'r') as f:
        content = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_cfg = RunConfig()
    model = neural_network_for_run(args=run_cfg, device=device)

    trainloader, valloader, class_weights = load_datasets(fold_id=4)

    loss_func = FocalLoss(alpha=class_weights[-1], gamma=run_cfg.focal_loss_gamma).to(device)

    for weight, folder in zip(weights, folders):

        assert(Path(weight).parent.parent.name == folder)

        print(f"Loading weights from {weight}")
        ckpt = torch.load(weight, map_location=device)
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()

        metrics = evaluate_on_generator(model, valloader, loss_func, run_cfg, device)

        print("\nFinal test‑fold metrics (metric units):")
        print(json.dumps(metrics, indent=2))

        output_file = Path('/home/zimon/flwr-picai-training/outputs/final_results') / folder / 'inference_results.json'

        if output_file is not None:
            with open(output_file, "w") as fp:
                json.dump(metrics, fp, indent=2)
            print(f"\nSaved metrics to {output_file}\n")


if __name__ == "__main__":
    main()
