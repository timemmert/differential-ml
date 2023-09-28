import torch
if torch.backends.mps.is_available():
    global_device = torch.device("mps")
else:
    global_device = torch.device("cpu")
