import torch
import math

if torch.cuda.is_available():
    # Check if CUDA is available (NVIDIA GPU)
    device = torch.device("cuda")
    device_name = "CUDA"
elif torch.backends.mps.is_available():
    # Check if the Apple GPU is available
    # this ensures that the current MacOS version is at least 12.3+
    device = torch.device("mps")
    device_name = "Apple GPU"
else:
    device = torch.device("cpu")
    device_name = "CPU"

print(f"\nPyTorch is using: {device_name}")

# this ensures that the current PyTorch installation was built with MPS activated (Apple GPU)

print ("\ntorch.backends.mps.is_built()", torch.backends.mps.is_built())

print ("")