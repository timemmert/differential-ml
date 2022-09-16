import torch


def uniform_random_in_range(
        *size: int,
        start: float = -10,
        end: float = 10,
        requires_grad: bool = True
) -> torch.tensor:
    if end <= start:
        raise ValueError("Start value for random number generation needs to be lower than end value.")
    return torch.rand(size, requires_grad=requires_grad) * torch.tensor((end - start)) + torch.tensor(start)
