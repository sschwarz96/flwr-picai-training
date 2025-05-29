# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

import torch

from src.picai_baseline.flwr.picai_client import client_fn
from src.picai_baseline.flwr.picai_server import server_fn
from src.picai_baseline.flwr.run_config import run_configuration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
backend_config = {"client_resources": {"num_cpus": run_configuration.num_threads_clients, "num_gpus": 0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config = {
        "client_resources": {"num_cpus": run_configuration.num_threads_clients,
                             "num_gpus": run_configuration.num_gpus}, }
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`

server = ServerApp(server_fn=server_fn)
client = ClientApp(client_fn=client_fn)

print(f"Total GPUs detected: {torch.cuda.device_count()}")
start_time = time.time()

# Run simulation
for x in range(1):
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=run_configuration.num_clients,
        backend_config=backend_config,
    )

end_time = time.time()
print(f"Execution time: {end_time - start_time:.3f} seconds")
directory = "/home/zimon/flwr-picai-training/outputs/2025-05-29"

# List only folders
folders = [os.path.join(directory, d) for d in os.listdir(directory)
           if os.path.isdir(os.path.join(directory, d))]

# Find the newest folder by modification time
newest_folder = max(folders, key=os.path.getmtime)

with open(f"{newest_folder}/fit_metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)
    metrics_dict = {"fit_metrics": metrics}
    metrics_dict["run_time"] = ( (end_time - start_time) / 60)

with open(f"{newest_folder}/fit_metrics.json", "w", encoding="utf-8") as fp:
    json.dump(metrics_dict, fp)
