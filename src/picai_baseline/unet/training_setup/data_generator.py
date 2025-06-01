# This file is part of the PiCAI Baseline U-Net (Apache 2.0 License)
# Modified by Simon Schwarz on 19.2.25 and 24.2.25
# Changes: Number of threads adjusted and function added that returns number of elements in dataloader

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

import json
from collections import OrderedDict
from pathlib import Path

import monai
import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from monai.transforms import Compose, EnsureType

from src.picai_baseline.unet.training_setup.image_reader import SimpleITKDataset


def default_collate(batch):
    """Collate multiple samples into batches of NumPy arrays in a dict."""
    sample = batch[0]

    # 1) If it’s already a NumPy scalar/array/list, stack with vstack or np.array
    if isinstance(sample, np.ndarray):
        return np.vstack(batch)
    if isinstance(sample, (int, np.int64)):
        return np.array(batch, dtype=np.int32)
    if isinstance(sample, (float, np.float32)):
        return np.array(batch, dtype=np.float32)
    if isinstance(sample, (np.float64,)):
        return np.array(batch, dtype=np.float64)

    # 2) If it’s a dict‐like, recurse per key
    if isinstance(sample, (dict, OrderedDict)):
        return {
            key: default_collate([d[key] for d in batch])
            for key in sample
        }

    # 3) If it’s a tuple/list, transpose and collate each field
    if isinstance(sample, (tuple, list)):
        transposed = list(zip(*batch))
        return [default_collate(field) for field in transposed]

    # 4) If it’s a torch.Tensor, first move to CPU & numpy, then vstack
    if isinstance(sample, torch.Tensor):
        # Move all to CPU & NumPy
        np_batch = [t.cpu().numpy() for t in batch]
        # Now each is a NumPy array, so we can vstack
        return np.vstack(np_batch)

    raise TypeError(f"Unknown batch element type: {type(sample)}")


class DataLoaderFromDataset(DataLoader):
    """Create dataloader from given dataset"""

    def __init__(self, data, batch_size, num_threads, seed_for_shuffle=1, collate_fn=default_collate,
                 return_incomplete=False, shuffle=True, infinite=False):
        super(DataLoaderFromDataset, self).__init__(data, batch_size, num_threads, seed_for_shuffle,
                                                    return_incomplete=return_incomplete, shuffle=shuffle,
                                                    infinite=infinite)
        self.collate_fn = collate_fn
        self.indices = np.arange(len(data))
        self.dataset = data

    def get_data_length(self) -> int:
        return len(self._data)

    def generate_train_batch(self):
        # randomly select N samples (N = batch size)
        indices = self.get_indices()

        # create dictionary per sample
        batch = [{'data': self._data[i][0].numpy(),
                  'seg': self._data[i][1].numpy()} for i in indices]

        return self.collate_fn(batch)


def prepare_datagens(args, fold_id):
    """Load data sheets --> Create datasets --> Create data loaders"""

    # load datasheets
    with open(Path(args.overviews_dir) / f'PI-CAI_val-fold-{fold_id}.json') as fp:
        train_json = json.load(fp)
    with open(Path(args.overviews_dir) / f'PI-CAI_val-fold-{fold_id}.json') as fp:
        valid_json = json.load(fp)

    # load paths to images and labels
    train_data = [np.array(train_json['image_paths']), np.array(train_json['label_paths'])]
    valid_data = [np.array(valid_json['image_paths']), np.array(valid_json['label_paths'])]

    # use case-level class balance to deduce required train-time class weights
    class_ratio_t = [int(np.sum(train_json['case_label'])), int(len(train_data[0]) - np.sum(train_json['case_label']))]
    class_ratio_v = [int(np.sum(valid_json['case_label'])), int(len(valid_data[0]) - np.sum(valid_json['case_label']))]
    class_weights = (class_ratio_t / np.sum(class_ratio_t))

    # log dataset definition
    print('Dataset Definition:', "-" * 80)
    print(f'Fold Number: {fold_id}')
    print('Data Classes:', list(np.unique(train_json['case_label'])))
    print(f'Train-Time Class Weights: {class_weights}')
    print(f'Training Samples [-:{class_ratio_t[1]};+:{class_ratio_t[0]}]: {len(train_data[1])}')
    print(f'Validation Samples [-:{class_ratio_v[1]};+:{class_ratio_v[0]}]: {len(valid_data[1])}')

    # dummy dataloader for sanity check
    pretx = [EnsureType()]
    check_ds = SimpleITKDataset(image_files=train_data[0][:args.virtual_batch_size * 2],
                                seg_files=train_data[1][:args.virtual_batch_size * 2],
                                transform=Compose(pretx),
                                seg_transform=Compose(pretx))
    check_loader = DataLoaderFromDataset(check_ds, batch_size=args.virtual_batch_size, num_threads=args.num_threads_clients)
    data_pair = monai.utils.misc.first(check_loader)
    print('DataLoader - Image Shape: ', data_pair['data'].shape)
    print('DataLoader - Label Shape: ', data_pair['seg'].shape)
    print("-" * 100)

    assert args.image_shape == list(data_pair['data'].shape[
                                    2:]), f"Expected shape {args.image_shape} but shape was {list(data_pair['data'].shape[2:])} "
    assert args.num_channels == data_pair['data'].shape[1]
    assert args.num_classes == len(np.unique(train_json['case_label']))

    # actual dataloaders used at train-time
    train_ds = SimpleITKDataset(image_files=train_data[0], seg_files=train_data[1],
                                transform=Compose(pretx), seg_transform=Compose(pretx))
    valid_ds = SimpleITKDataset(image_files=valid_data[0], seg_files=valid_data[1],
                                transform=Compose(pretx), seg_transform=Compose(pretx))
    train_ldr = DataLoaderFromDataset(train_ds,
                                      batch_size=args.virtual_batch_size, num_threads=args.num_threads_clients, infinite=True,
                                      shuffle=True)
    valid_ldr = DataLoaderFromDataset(valid_ds,
                                      batch_size=args.virtual_batch_size, num_threads=args.num_threads_clients, infinite=False,
                                      shuffle=False)

    return train_ldr, valid_ldr, class_weights.astype(np.float32)
