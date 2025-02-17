import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Total GPUs detected: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")