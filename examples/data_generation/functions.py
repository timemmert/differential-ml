from enum import Enum
from typing import Callable, Tuple

import numpy as np
import torch
from torch.autograd.functional import jacobian

from examples.data_generation.random_data import uniform_random_in_range


class OutputFormat(Enum):
    TORCH = 0
    TF = 1
    NUMPY = 2


def aad_datagen(
        x: Tuple[torch.tensor, np.array],
        datagen_function: Callable[[torch.tensor], torch.tensor],
        output_format: OutputFormat = OutputFormat.NUMPY
):
    if not torch.is_tensor(x):
        x = torch.tensor(x, requires_grad=True)
    if not x.requires_grad:
        x.requires_grad_(True)

    y = datagen_function(x)
    """
    The following line is not ideal, the reason is discussed here:
    https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571
    
    As the input x is of shape (n, n_x) and y of shape (n, n_y), dydx is computed as (n, n_y, n, n_x). As n becomes
    bigger, the computation becomes slower due to the differentiation in the unnecessary dimension.
    """
    dydx = jacobian(datagen_function, x).sum(dim=2)

    if output_format == OutputFormat.NUMPY:
        return x.detach().numpy(), y.detach().numpy(), dydx.detach().numpy()
    elif output_format == OutputFormat.TORCH:
        return x, y, dydx
    else:
        raise NotImplementedError(f"Output format {output_format} not supported")


def trigonometric_one_in_one_out(n: int, x_range: Tuple[float, float] = (-10, 10),
                                 output_format: OutputFormat = OutputFormat.NUMPY):
    x = uniform_random_in_range(n, 1, start=x_range[0], end=x_range[1])
    return aad_datagen(x, lambda x_: (torch.sin(x_[:, 0]) + x_[:, 0] * torch.cos(x_[:, 0]))[:, None], output_format=output_format)


def quadratic_one_in_one_out(a: float, n: int, x_range: Tuple[float, float] = (-10, 10),
                             output_format: OutputFormat = OutputFormat.NUMPY):
    x = uniform_random_in_range(n, 1, start=x_range[0], end=x_range[1])
    return aad_datagen(x, lambda x_: (a * x_[:, 0] ** 2)[:, None], output_format=output_format)


def quadratic_two_in_one_out(a: float, n: int, x_range: Tuple[float, float] = (-10, 10),
                             output_format: OutputFormat = OutputFormat.NUMPY):
    x = uniform_random_in_range(n, 2, start=x_range[0], end=x_range[1])
    return aad_datagen(x, lambda x_: (a * x_[:, 0] ** 2 * x_[:, 1] ** 2)[:, None], output_format=output_format)


def polynomial_trigonometric_two_in_one_out(n: int, x_range: Tuple[float, float] = (-10, 10),
                                            output_format: OutputFormat = OutputFormat.NUMPY):
    x = uniform_random_in_range(n, 2, start=x_range[0], end=x_range[1])
    return aad_datagen(x, lambda x_: (x_[:, 0] ** 3 * x_[:, 1] ** 2 + torch.sin(x_[:, 0]) * x_[:, 1])[:, None],
                       output_format=output_format)


def trigonometric_two_in_two_out(
        n: int,
        x_range: Tuple[float, float] = (-10, 10),
        output_format: OutputFormat = OutputFormat.NUMPY
):
    x = uniform_random_in_range(n, 2, start=x_range[0], end=x_range[1])
    return aad_datagen(
        x,
        lambda x_: torch.stack(
            (
                torch.sin(x_[:, 0]), torch.cos(x_[:, 1])
            ),
            dim=1
        ),
        output_format=output_format
    )
