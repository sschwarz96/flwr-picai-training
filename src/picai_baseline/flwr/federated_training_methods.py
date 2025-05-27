import os
from collections import OrderedDict
from logging import INFO
from pathlib import Path
from typing import List

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, Parameters, log
from torch import nn

from src.picai_baseline.flwr.run_config import run_configuration
from src.picai_baseline.unet.training_setup.callbacks import optimize_model, validate_model
from src.picai_baseline.unet.training_setup.data_generator import prepare_datagens


def load_datasets(fold_id: int):
    return prepare_datagens(args=run_configuration, fold_id=fold_id)


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def train(model, optimizer, loss_func, train_gen, arguments, device,
          training_epochs):
    train_loss = np.inf
    norms =[]
    for i in range(training_epochs):
        model.train()
        model, optimizer, train_gen, train_loss, all_norms = optimize_model(
            model=model, optimizer=optimizer, loss_func=loss_func, train_gen=train_gen,
            args=arguments, device=device, epoch=i
        )
        norms.append(all_norms)
    norms = torch.cat(norms)
    return train_loss,norms


def test(model, optimizer, loss_func, valid_gen, arguments, device):
    model.eval()
    with torch.no_grad():  # no gradient updates during validation

        model, optimizer, valid_gen, tracking_metrics = validate_model(
            model=model, optimizer=optimizer, loss_func=loss_func, valid_gen=valid_gen, args=arguments,
            device=device
        )

    return tracking_metrics



def load_model_checkpoint(net: nn.Module) -> Parameters:
    outputs = Path("./outputs")
    list_of_files = [fname for fname in outputs.rglob("*.pth")]
    latest_round_file = max(list_of_files, key=os.path.getctime)
    latest_round_file = Path(
        '//home/zimon/flwr-picai-training/outputs/final_results/DA/no_DP_DA_enabled/14-45-33/model_state_rank_0.310247162239236_round_1.pth')
    log(INFO, f"Loading pre-trained model from: {latest_round_file}")
    state_dict = torch.load(latest_round_file)
    net.load_state_dict(state_dict)
    state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
    parameters = ndarrays_to_parameters(state_dict_ndarrays)
    return parameters
