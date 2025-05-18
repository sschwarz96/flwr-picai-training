import torch
from opacus import PrivacyEngine
from torchviz import make_dot

from picai_baseline.flwr.run_config import RunConfig
import torch
from opacus import PrivacyEngine
from torch.utils.tensorboard import SummaryWriter
# Import run config class and neural network builder
from picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run
from picai_baseline.unet.training_setup.compute_spec import compute_spec_for_run

# Initialize configuration and computation device
run_configuration = RunConfig()
device = compute_spec_for_run()

# Initialize privacy engine (optional here, unless training with DP)
privacy_engine = PrivacyEngine(accountant="rdp")

# Build the model using your configuration
model = neural_network_for_run(args=run_configuration, device=device)

# Make sure the model is on the correct device
model.to(device)

# Prepare a dummy input for the model
# Your model expects input shape [batch_size, channels, depth, height, width]
dummy_input = torch.randn(8, 3, 16, 128, 128, device=device, requires_grad=True)

simplified_params = {
    name: param for name, param in model.named_parameters()
    if "up" in name or "down" in name or "out" in name
}
# Forward pass
output = model(dummy_input)

# Create visualization
dot = make_dot(output, params=simplified_params)
dot.format = "png"  # or 'png', 'svg'
dot.render("monai_unet_detailed_graph")
