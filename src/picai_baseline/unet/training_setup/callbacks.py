# This file is part of the PiCAI Baseline U-Net (Apache 2.0 License)
# Modified by Simon Schwarz on 19.2.25
# Changes: Adapted to also track validation loss and removed writer logic + resume/restart logic
import itertools
#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import time
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from picai_eval import Metrics
from picai_eval.eval import evaluate_case
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter

from src.picai_baseline.unet.training_setup.poly_lr import poly_lr


def optimize_model(model, optimizer, loss_func, train_gen, args, device, epoch):
    """Optimize model x N training steps per epoch + update learning rate"""

    train_loss = 0.0
    start_time = time.time()
    steps_done = 0
    total_steps = ceil(train_gen.data_loader.get_data_length() / args.batch_size)

    # 2) Take exactly steps_per_epoch batches from train_gen
    for batch_data in train_gen:
        steps_done += 1
        # unpack list→dict if needed
        if isinstance(batch_data, list):
            batch_data = {"data": batch_data[0], "seg": batch_data[1]}

        # move to device
        inputs = torch.as_tensor(batch_data["data"], device=device)
        labels = torch.as_tensor(batch_data["seg"], device=device)
        labels = fix_labels_shape(args, labels)

        # forward + loss
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        train_loss += loss.item()

        # backpropagate + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps_done == total_steps:
            break

    # 3) Update learning rate (your poly schedule)
    updated_lr = poly_lr(
        epoch + 1,
        args.num_train_epochs,
        args.base_lr,
        min_lr=1e-6,  # a tiny floor
        exponent=0.95  # your intended decay power
    )
    optimizer.param_groups[0]['lr'] = updated_lr
    print(f"Learning Rate Updated! New Value: {np.round(updated_lr, 10)}", flush=True)

    # 4) Log epoch summary
    avg_loss = train_loss / steps_done
    elapsed = int(time.time() - start_time)
    print("-" * 100)
    print(
        f"Epoch {epoch + 1}/{args.num_train_epochs} "
        f"(Train. Loss: {avg_loss:.4f}; Time: {elapsed} sec; "
        f"Steps: {steps_done})", flush=True
    )

    return model, optimizer, train_gen, avg_loss


def fix_labels_shape(args, labels):
    # bugfix for shape of targets
    if labels.shape[1] == 1:
        # labels now has shape (B, 1, D, H, W)
        labels = labels[:, 0, ...]  # shape: (B, D, H, W)
        labels = F.one_hot(labels.long(), num_classes=args.num_classes).float()  # shape: (B, D, H, W, C)
        # reshape to (B, C, D, H, W)
        labels = labels.permute(0, 4, 1, 2, 3).contiguous()
    return labels


def validate_model(model, optimizer, loss_func, valid_gen, args, device):
    """Validate model per N epoch + export model weights"""

    # epoch, f = tracking_metrics['epoch'], tracking_metrics['fold_id']

    # for each validation sample
    lesion_results = []
    total_loss = 0.0  # Track loss
    step = 0
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
        valid_labels_device = fix_labels_shape(args, valid_labels_device)

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
    return model, optimizer, valid_gen, tracking_metrics
