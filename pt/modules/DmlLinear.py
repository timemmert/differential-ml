from typing import Callable

import torch
from torch import nn, autograd

from pt.modules.DmlModule import DmlModule


class DmlLinear(DmlModule):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            activation: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

        self.activation = activation

    def activation_derivative(self, x: torch.Tensor) -> torch.Tensor:
        # noinspection PyTypeChecker
        return torch.diagonal(autograd.functional.jacobian(self.activation, x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.activation(x) @ self.weight) + self.bias

    def greek(self, x: torch.Tensor, prev_greek: torch.Tensor) -> torch.Tensor:
        """
        For a given forward pass with input `x`, i.e. `y = forward(x)`; we can compute the greek of the output.
        """
        return (prev_greek @ self.weight.T) * self.activation_derivative(x)
