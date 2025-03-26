"""pytorch-example: A Flower / PyTorch app."""

import json
from datetime import datetime
from logging import INFO
from pathlib import Path
from typing import Optional, Union

import torch

from flwr.common import logger, parameters_to_ndarrays, EvaluateRes
from flwr.common.typing import UserConfig, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.picai_baseline.flwr.federated_training_methods import set_parameters
from src.picai_baseline.flwr.run_config import run_configuration
from src.picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run

PROJECT_NAME = "FLOWER-advanced-pytorch"


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found, (3) logs
    results to W&B if enabled.
    """

    def __init__(self, run_config: UserConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        # Keep track of best acc
        self.best_ranking_so_far = 0.0
        # A dictionary to store results as they come
        self.results = {}
        # Store latest params
        self.params = None

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_ranking(self, round, ranking):
        """Determines if a new best global model has been found.

        If so, the model checkpoint is saved to disk.
        """
        if ranking > self.best_ranking_so_far and round != 0:
            self.best_ranking_so_far = ranking
            logger.log(INFO, "💡 New best global model found: %f", ranking)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays
            ndarrays = parameters_to_ndarrays(self.params)
            model = neural_network_for_run(args=run_configuration)
            set_parameters(model, ndarrays)
            # Save the PyTorch model
            file_name = f"model_state_rank_{ranking}_round_{round}.pth"
            torch.save(model.state_dict(), self.save_path / file_name)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:

        print(f"SERVER ROUND {server_round}")
        if server_round == 0 and not run_configuration.resume_training:
            return None
        loss, metrics = super().evaluate(server_round=server_round, parameters=parameters)
        self._update_best_ranking(server_round, metrics["ranking"])

        self.store_results_and_log(
            server_round=server_round,
            tag="central_evaluate",
            results_dict={"central_evaluate_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        self.params = aggregated_parameters

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results: list[tuple[ClientProxy, EvaluateRes]], failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self._update_best_ranking(server_round, metrics["ranking"])

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir
