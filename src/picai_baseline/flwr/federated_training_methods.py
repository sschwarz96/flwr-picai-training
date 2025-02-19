import os
import threading
from random import randint

from src.picai_baseline.unet.training_setup.augmentations.nnUNet_DA import \
    apply_augmentations
from src.picai_baseline.unet.training_setup.callbacks import (
    resume_or_restart_training)
from src.picai_baseline.unet.training_setup.compute_spec import \
    compute_spec_for_run
from src.picai_baseline.unet.training_setup.data_generator import prepare_datagens
from src.picai_baseline.unet.training_setup.loss_functions.focal import FocalLoss
from src.picai_baseline.unet.training_setup.neural_network_selector import \
    neural_network_for_run
from torch.utils.tensorboard import SummaryWriter

import torch
import flwr
from flwr.client import Client
from flwr.common import Context, Metrics
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

from src.picai_baseline.flwr.PicaiClient import PicaiFlowerCLient

NUM_CLIENTS = 5
BATCH_SIZE = 4

# Global counter and lock to assign unique client IDs across invocations of client_fn
_client_counter_lock = threading.Lock()
_client_counter = 0


class Args:
    # Data I/O + Experimental Setup
    max_threads = 6
    validate_n_epochs = 3
    validate_min_epoch = 0
    export_best_model = 1
    resume_training = 0  # Assuming it should be an int, change to str if needed
    weights_dir = "/home/zimon/picai_baseline/workdir/results/UNet/weights"  # Required, default to an empty string
    overviews_dir = "/home/zimon/picai_baseline/workdir/results/UNet/overviews/Task2203_picai_baseline"  # Required, default to an empty string
    folds = [0, 1, 2, 3, 4]  # Assuming a list of integers

    # Training Hyperparameters
    image_shape = [20, 256, 256]  # (z, y, x)
    num_channels = 3
    num_classes = 2
    num_epochs = 5
    base_lr = 0.001
    focal_loss_gamma = 1.0
    enable_da = 1  # Data Augmentation

    # Neural Network-Specific Hyperparameters
    model_type = "unet"
    model_strides = [(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]  # Converted from string
    model_features = [32, 64, 128, 256, 512, 1024]  # Converted from string
    batch_size = BATCH_SIZE
    use_def_model_hp = 1


arguments = Args()


def load_datasets(fold_id: int):
    return prepare_datagens(args=arguments, fold_id=fold_id)


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]

    random_number = randint(0,NUM_CLIENTS)

    fold_id = (partition_id + random_number) % NUM_CLIENTS


    # Call compute_spec_for_run with assigned GPU
    device, args = compute_spec_for_run(args=arguments)

    print(f"Initial Device type: {type(device)}, Value: {device}")

    trainloader, valloader, class_weights = load_datasets(fold_id=fold_id)

    train_gen = apply_augmentations(
        dataloader=trainloader,
        num_threads=args.max_threads,
        disable=(not bool(args.enable_da))
    )

    # initialize multi-threaded augmenter in background
    train_gen.restart()

    # Load model
    net = neural_network_for_run(args=args, device=device)

    # loss function + optimizer
    loss_func = FocalLoss(alpha=class_weights[-1], gamma=args.focal_loss_gamma).to(device)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.base_lr, amsgrad=True)

    writer = SummaryWriter()

    model, optimizer, tracking_metrics = resume_or_restart_training(
        model=net, optimizer=optimizer,
        device=device, args=args, fold_id=partition_id
    )

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return PicaiFlowerCLient(net, train_gen, valloader, optimizer, loss_func, args, device, partition_id).to_client()


def fit_config(server_round: int):
    """Generate training configuration for each round."""
    # Create the configuration dictionary
    config = {
        "current_round": server_round,
        "local_epochs": arguments.num_epochs,
    }
    return config


def evaluate_config(server_round: int):
    """Generate evaluation configuration for each round."""
    # Create the configuration dictionary
    config = {
        "current_round": server_round,
        "local_epochs": 1,
    }
    return config


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average for all relevant metrics, excluding loss."""
    weighted_metrics = {}
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {key: 0 for key in metrics[0][1] if key != "loss"}  # Avoid division by zero

    for key in metrics[0][1]:  # Iterate over metric keys
        if key == "loss":  # Skip loss
            continue

        weighted_values = [num_examples * m[key] for num_examples, m in metrics]
        weighted_metrics[key] = sum(weighted_values) / total_examples

    return weighted_metrics


strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=1,  # Never sample less than 10 clients for training
    min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    on_fit_config_fn=fit_config,  # Pass the `fit_config` function
    on_evaluate_config_fn=evaluate_config,
    evaluate_metrics_aggregation_fn=weighted_average,
)


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)

    return ServerAppComponents(strategy=strategy, config=config)
