from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.cuda
from flwr.client import NumPyClient
from picai_baseline.unet.training_setup.callbacks import optimize_model, validate_model


class PicaiFlowerCLient(NumPyClient):
    def __init__(self, net, trainloader, valloader, optimizer, loss_func, tracking_metrics, arguments,
                 device, writer):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.tracking_metrics = tracking_metrics
        self.args = arguments
        self.device = device
        self.writer = writer

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        print(f"Device type: {type(self.device)}, Value: {self.device}")
        self.tracking_metrics = train(self.net, self.optimizer, self.loss_func, self.trainloader, self.args,
                                      self.tracking_metrics, self.device,
                                      self.writer, config["local_epochs"])
        return get_parameters(self.net), self.trainloader.generator.get_data_length(), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, average_precision = test(self.net, self.optimizer, self.valloader, self.args,
                                       self.tracking_metrics, self.device,
                                       self.writer, config["local_epochs"])
        return float(loss), self.valloader.get_data_length(), {"Average Precision": float(average_precision)}


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def train(model, optimizer, loss_func, train_gen, arguments, tracking_metrics, device, writer,
          training_epochs):
    for i in range(training_epochs):
        model.train()
        tracking_metrics['epoch'] = i

        model, optimizer, train_gen, tracking_metrics, writer = optimize_model(
            model=model, optimizer=optimizer, loss_func=loss_func, train_gen=train_gen,
            args=arguments, tracking_metrics=tracking_metrics, device=device, writer=writer
        )
    print(tracking_metrics)
    return tracking_metrics


def test(model, optimizer, valid_gen, arguments, tracking_metrics, device, writer, training_epochs):
    model.eval()
    with torch.no_grad():  # no gradient updates during validation

        model, optimizer, valid_gen, tracking_metrics, writer = validate_model(
            model=model, optimizer=optimizer, valid_gen=valid_gen, args=arguments,
            tracking_metrics=tracking_metrics, device=device, writer=writer

        )

    return tracking_metrics["all_train_loss"][training_epochs], tracking_metrics["all_valid_metrics_ap"][
        training_epochs],
