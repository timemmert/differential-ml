from enum import Enum
from typing import Callable, Tuple

import torch

from examples.data_generation.random_data import uniform_random_in_range


class OutputFormat(Enum):
    TORCH = 0
    TF = 1
    NUMPY = 2


def aad_datagen(x: torch.tensor, datagen_function: Callable[[torch.tensor], torch.tensor], output_format: OutputFormat = OutputFormat.NUMPY):
    y = datagen_function(x)

    x.retain_grad()
    y.backward(torch.ones(x.shape[0]))
    dydx = x.grad

    y = y.reshape(-1, 1)
    if output_format == OutputFormat.NUMPY:
        return x.detach().numpy(), y.detach().numpy(), dydx.detach().numpy()
    elif output_format == OutputFormat.TORCH:
        return x, y, dydx
    else:
        raise NotImplementedError(f"Output format {output_format} not supported")


def trigonometric_one_dim(n: int, x_range: Tuple[float, float] = (-10, 10), output_format: OutputFormat = OutputFormat.NUMPY):
    x = uniform_random_in_range(n, 1, start=x_range[0], end=x_range[1])
    return aad_datagen(x, lambda x_: torch.sin(x_[:, 0]) + x_[:, 0] * torch.cos(x_[:, 0]), output_format=output_format)


def quadratic_one_dim(a: float, n: int, x_range: Tuple[float, float] = (-10, 10), output_format: OutputFormat = OutputFormat.NUMPY):
    x = uniform_random_in_range(n, 1, start=x_range[0], end=x_range[1])
    return aad_datagen(x, lambda x_: a * x_[:, 0] ** 2, output_format=output_format)


def quadratic_two_dim(a: float, n: int, x_range: Tuple[float, float] = (-10, 10), output_format: OutputFormat = OutputFormat.NUMPY):
    x = uniform_random_in_range(n, 2, start=x_range[0], end=x_range[1])
    return aad_datagen(x, lambda x_: a * x_[:, 0] ** 2 * x_[:, 1] ** 2, output_format=output_format)


def polynomial_trigonometric_two_dim(n: int, x_range: Tuple[float, float] = (-10, 10), output_format: OutputFormat = OutputFormat.NUMPY):
    x = uniform_random_in_range(n, 2, start=x_range[0], end=x_range[1])
    return aad_datagen(x, lambda x_: x_[:, 0] ** 3 * x_[:, 1] ** 2 + torch.sin(x_[:, 0]) * x_[:, 1], output_format=output_format)
