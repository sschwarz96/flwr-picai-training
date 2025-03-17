# This file is part of the PiCAI Baseline U-Net (Apache 2.0 License)
# Modified by Simon Schwarz on 19.2.25
# Changes: Adapted to also track validation loss and removed writer logic


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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from picai_eval import Metrics
from picai_eval.eval import evaluate_case
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter

from src.picai_baseline.unet.training_setup.poly_lr import poly_lr


def optimize_model(model, optimizer, loss_func, train_gen, args, device, epoch):
    """Optimize model x N training steps per epoch + update learning rate"""

    train_loss, step = 0, 0
    start_time = time.time()
    # epoch = tracking_metrics['epoch']

    # for each mini-batch or optimization step
    for batch_data in train_gen:
        step += 1
        try:
            inputs = batch_data['data'].to(device)
            labels = batch_data['seg'].to(device)
        except Exception as e:
            inputs = torch.from_numpy(batch_data['data']).to(device)
            labels = torch.from_numpy(batch_data['seg']).to(device)

        labels = fix_labels_shape(args, labels)

        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        train_loss += loss.item()

        # backpropagate + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # define each training epoch == 100 steps (note: nnU-Net uses 250 steps)
        if step >= 100:
            break

    # update learning rate
    updated_lr = poly_lr(epoch + 1, args.num_train_epochs, args.base_lr, 0.95)
    optimizer.param_groups[0]['lr'] = updated_lr
    print("Learning Rate Updated! New Value: " + str(np.round(updated_lr, 10)), flush=True)

    # track training metrics
    train_loss /= step
    # tracking_metrics['train_loss'] = train_loss
    # writer.add_scalar("train_loss", train_loss, epoch + 1)
    print("-" * 100)
    print(f"Epoch {epoch + 1}/{args.num_train_epochs} (Train. Loss: {train_loss:.4f}; \
        Time: {int(time.time() - start_time)}sec; Steps Completed: {step})", flush=True)

    return model, optimizer, train_gen, train_loss


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

    tracking_metrics = {
        "average_precision": valid_metrics.AP,
        "auroc": valid_metrics.auroc,
        "ranking": valid_metrics.score,
        "loss": avg_loss  # Include loss
    }
    return model, optimizer, valid_gen, tracking_metrics
