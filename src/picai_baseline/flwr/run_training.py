# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
import json
from pathlib import Path

from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

import torch

from src.picai_baseline.flwr.picai_client import client_fn
from src.picai_baseline.flwr.picai_server import server_fn
from src.picai_baseline.flwr.run_config import run_configuration

config_path = Path('/home/zimon/flwr-picai-training/src/picai_baseline/flwr/run_config.json')
EPSILON = [20, 30, 20, 30]

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
# Run simulation
for x in range(len(EPSILON)):
    # Read, modify, and write JSON safely
    with open(config_path, 'r') as f:
        data = json.load(f)

    data['epsilon'] = EPSILON[x]  # or any other update

    if x == 2:
        data['enable_da'] = False

    with open(config_path, 'w') as f:
        json.dump(data, f, indent=4)
    for i in range(3):
        run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=run_configuration.num_clients,
            backend_config=backend_config,
        )
