import shutil

import numpy as np
from flwr.common import Context, Metrics, log, NDArrays, RecordSet
from flwr.server import ServerAppComponents, ServerConfig

from src.picai_baseline.flwr.custom_strategy import CustomFedAvg
from src.picai_baseline.flwr.federated_training_methods import load_model_checkpoint, set_parameters, test, \
    load_datasets
from src.picai_baseline.flwr.run_config import run_configuration
from src.picai_baseline.unet.training_setup.compute_spec import compute_spec_for_run
from src.picai_baseline.unet.training_setup.loss_functions.focal import FocalLoss
from src.picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run
import torch
from flwr.common.typing import Scalar


def fit_config(server_round: int):
    """Generate training configuration for each round."""
    # Create the configuration dictionary
    config = {
        "current_round": server_round,
        "local_epochs": run_configuration.num_train_epochs,
        "ttl": 3600
    }
    return config


def average_privacy(
        metrics_list: list[tuple[int, dict[str, Scalar]]]
) -> dict[str, Scalar]:
    # metrics_list is [(num_examples, {"epsilon":..., "alpha":...}), ...]
    epsilons = [metrics["epsilon"] for _, metrics in metrics_list]
    alphas = [metrics["alpha"] for _, metrics in metrics_list]
    return {
        "epsilon": float(np.mean(epsilons)),
        "alpha": float(np.mean(alphas)),
    }


def evaluate_config(server_round: int):
    """Generate evaluation configuration for each round."""
    # Create the configuration dictionary
    config = {
        "current_round": server_round,
        "local_epochs": run_configuration.validate_n_epochs,
        "tll": 3600
    }
    return config


def create_central_evaluation():
    """Create evaluation components for centralized evaluation."""
    device = compute_spec_for_run()

    # Load centralized test dataset
    # Note: You might want to use a specific fold or combine validation sets
    _, test_loader, class_weights = load_datasets(fold_id=run_configuration.evaluation_fold)

    # Initialize model for evaluation
    net = neural_network_for_run(args=run_configuration, device=device)
    loss_func = FocalLoss(alpha=class_weights[-1], gamma=run_configuration.focal_loss_gamma).to(device)

    return net, test_loader, loss_func, device, run_configuration


def centralized_evaluate(
        server_round: int,
        parameters: NDArrays,
        config: dict,
) -> tuple[float, dict]:
    """Centralized evaluation function to be used by the server."""
    net, test_loader, loss_func, device, args = create_central_evaluation()

    # Update model with the latest parameters
    set_parameters(net, parameters)
    net.eval()

    with torch.no_grad():
        tracking_metrics = test(
            model=net,
            optimizer=None,  # Not needed for evaluation
            loss_func=loss_func,
            valid_gen=test_loader,
            arguments=args,
            device=device
        )

    # Extract loss and other metrics
    loss = float(tracking_metrics.pop("loss"))

    return loss, tracking_metrics


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
    try:
        shutil.rmtree('/home/zimon/.flwr_dp_states')
    except OSError:
        print("No dp states from before")

    strategy = CustomFedAvg(
        run_config=run_configuration.to_dict(),
        fraction_fit=run_configuration.fraction_fit,
        fraction_evaluate=run_configuration.evaluate_fit,
        min_available_clients=run_configuration.num_clients,  # Wait until all 10 clients are available
        on_fit_config_fn=fit_config,  # Pass the `fit_config` function
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=average_privacy,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    if run_configuration.central_evaluation:
        strategy.evaluate_fn = centralized_evaluate
    else:
        strategy.on_evaluate_config_fn = evaluate_config,
        strategy.evaluate_metrics_aggregation_fn = weighted_average

    if run_configuration.resume_training:
        init_parameters = load_model_checkpoint(neural_network_for_run(args=run_configuration))
        strategy.initial_parameters = init_parameters

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=run_configuration.num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
