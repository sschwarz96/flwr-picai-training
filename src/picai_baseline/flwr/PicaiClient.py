from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.cuda
from flwr.client import NumPyClient
from src.picai_baseline.unet.training_setup.callbacks import optimize_model, validate_model


class PicaiFlowerCLient(NumPyClient):
    def __init__(self, net, trainloader, valloader, optimizer, loss_func, arguments,
                 device, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.args = arguments
        self.device = device
        self.partition_id = partition_id

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        print(f"Device type: {type(self.device)}, Value: {self.device}")
        train(self.net, self.optimizer, self.loss_func, self.trainloader, self.args, self.device,
              config["local_epochs"])
        return get_parameters(self.net), self.trainloader.generator.get_data_length(), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        tracking_metrics = test(self.net, self.optimizer, self.loss_func, self.valloader, self.args,
                                self.device)
        loss = float(tracking_metrics["loss"])
        tracking_metrics.popitem()

        return loss, self.valloader.get_data_length(), tracking_metrics


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def train(model, optimizer, loss_func, train_gen, arguments, device,
          training_epochs):
    train_loss = np.inf
    for i in range(training_epochs):
        model.train()
        model, optimizer, train_gen, train_loss = optimize_model(
            model=model, optimizer=optimizer, loss_func=loss_func, train_gen=train_gen,
            args=arguments, device=device, epoch=i
        )
    return train_loss


def test(model, optimizer, loss_func, valid_gen, arguments, device):
    model.eval()
    with torch.no_grad():  # no gradient updates during validation

        model, optimizer, valid_gen, tracking_metrics = validate_model(
            model=model, optimizer=optimizer, loss_func=loss_func, valid_gen=valid_gen, args=arguments,
            device=device
        )

    return tracking_metrics
