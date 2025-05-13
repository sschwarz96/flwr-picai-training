import random
from random import randint

import torch
import torch.cuda
from flwr.client import NumPyClient, Client
from flwr.common import Context

from src.picai_baseline.flwr.federated_training_methods import load_datasets, set_parameters, \
    get_parameters, train, test
from src.picai_baseline.flwr.run_config import run_configuration
from src.picai_baseline.unet.training_setup.augmentations.nnUNet_DA import apply_augmentations
from src.picai_baseline.unet.training_setup.compute_spec import compute_spec_for_run
from src.picai_baseline.unet.training_setup.loss_functions.focal import FocalLoss
from src.picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run


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
        print(f"Partition id {self.partition_id}")
        set_parameters(self.net, parameters)
        print(f"Device type: {type(self.device)}, Value: {self.device}")
        train(self.net, self.optimizer, self.loss_func, self.trainloader, self.args, self.device,
              config["local_epochs"])
        return get_parameters(self.net), self.trainloader.data_loader.get_data_length(), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        tracking_metrics = test(self.net, self.optimizer, self.loss_func, self.valloader, self.args,
                                self.device)
        loss = float(tracking_metrics["loss"])
        tracking_metrics.popitem()

        return loss, self.valloader.get_data_length(), tracking_metrics


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]

    # random.seed(run_configuration.random_seed)
    # random_number = randint(0, len(run_configuration.folds))

    fold_id = partition_id % len(run_configuration.folds)

    # Call compute_spec_for_run with assigned GPU
    device, args = compute_spec_for_run(args=run_configuration)

    print(f"Initial Device type: {type(device)}, Value: {device}")

    trainloader, valloader, class_weights = load_datasets(fold_id=fold_id)

    print(f"Valloader length: {valloader.get_data_length()}")

    train_gen = apply_augmentations(
        dataloader=trainloader,
        num_threads=run_configuration.num_threads_augmenting,
        disable=(not bool(run_configuration.enable_da))
    )

    if run_configuration.num_threads_augmenting > 1:
        # initialize multi-threaded augmenter in background
        train_gen.restart()

    # Load model
    net = neural_network_for_run(args=args, device=device)

    # loss function + optimizer
    loss_func = FocalLoss(alpha=class_weights[-1], gamma=args.focal_loss_gamma).to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.base_lr, amsgrad=True, weight_decay=1e-4)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return PicaiFlowerCLient(net, train_gen, valloader, optimizer, loss_func, args, device, partition_id).to_client()
