from typing import Tuple

import torch


def uniform_random_in_range(
    *size: int,
    data_range: Tuple[float, float] = (-10, 10)
) -> torch.tensor:
    start = data_range[0]
    end = data_range[1]
    if end <= start:
        raise ValueError(
            "Start value for random number generation needs to be lower than end value."
        )
    return torch.rand(size) * torch.tensor((end - start)) + torch.tensor(start)
