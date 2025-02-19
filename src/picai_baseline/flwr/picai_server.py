from logging import INFO
from symbol import parameters

import torch
from batchgenerators.dataloading.data_loader import DataLoader
from flwr.common import Context, Metrics, log
from flwr.server import ServerAppComponents, ServerConfig

from src.picai_baseline.flwr.custom_strategy import CustomFedAvg
from src.picai_baseline.flwr.federated_training_methods import load_model_checkpoint, set_parameters, test
from src.picai_baseline.flwr.run_config import run_configuration
from src.picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run


def fit_config(server_round: int):
    """Generate training configuration for each round."""
    # Create the configuration dictionary
    config = {
        "current_round": server_round,
        "local_epochs": run_configuration.num_train_epochs,
    }
    return config


def evaluate_config(server_round: int):
    """Generate evaluation configuration for each round."""
    # Create the configuration dictionary
    config = {
        "current_round": server_round,
        "local_epochs": run_configuration.validate_n_epochs,
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


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    if run_configuration.resume_training:
        init_parameters = load_model_checkpoint(neural_network_for_run(args=run_configuration))
        strategy = CustomFedAvg(
            run_config=run_configuration.to_dict(),
            initial_parameters=init_parameters,
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=1,  # Never sample less than 10 clients for training
            min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
            min_available_clients=run_configuration.num_clients,  # Wait until all 10 clients are available
            on_fit_config_fn=fit_config,  # Pass the `fit_config` function
            on_evaluate_config_fn=evaluate_config,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        strategy = CustomFedAvg(
            run_config=run_configuration.to_dict(),
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
            min_fit_clients=1,  # Never sample less than 10 clients for training
            min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
            min_available_clients=run_configuration.num_clients,  # Wait until all 10 clients are available
            on_fit_config_fn=fit_config,  # Pass the `fit_config` function
            on_evaluate_config_fn=evaluate_config,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=run_configuration.num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
