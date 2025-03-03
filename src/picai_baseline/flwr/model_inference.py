import torch

from src.picai_baseline.flwr.run_config import run_configuration
from src.picai_baseline.unet.training_setup.compute_spec import compute_spec_for_run
from src.picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run

device, args = compute_spec_for_run(args=run_configuration)

# 1. Initialize your model architecture
model = neural_network_for_run(args=args, device=device)

# 2. Load the model weights from checkpoint
checkpoint = torch.load('path/to/your/checkpoint.pth')

# 3. Load weights into your model (handle different checkpoint formats)
# Option A: If the checkpoint contains only the model state dict
model.load_state_dict(checkpoint)

# Option B: If the checkpoint contains more info (common for training checkpoints)
model.load_state_dict(checkpoint['model_state_dict'])  # Adjust key as needed

# 4. Set model to evaluation mode (important!)
model.eval()

# 5. Prepare your new data
from torch.utils.data import DataLoader

test_dataset = YourDataset('path/to/new/data')
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 6. Run inference
results = []
with torch.no_grad():  # Disable gradient calculation for inference
    for data in test_loader:
        # Prepare inputs (adjust according to your data format)
        inputs = data['input'].to(device)  # Move to GPU if available

        # Forward pass
        outputs = model(inputs)

        # Process outputs as needed
        predictions = torch.argmax(outputs, dim=1)  # Example for classification
        results.append(predictions)

# 7. Process and use your results
# ...