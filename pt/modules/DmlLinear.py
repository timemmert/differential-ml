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
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.empty(output_dim))

        self.activation: Callable[[torch.Tensor], torch.Tensor] = activation

        self._initialize_parameters()

    def _initialize_parameters(self):
        if self.activation == nn.ReLU:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
            self.bias.data.fill_(0.01)
        elif self.activation == nn.LeakyReLU:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity='leaky_relu')
            self.bias.data.fill_(0.01)
        else:
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0)

    def activation_derivative(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_reshaped = x.flatten()
        # noinspection PyTypeChecker
        _, jacobian_reshape = autograd.functional.vjp(
            func=self.activation,
            inputs=x_reshaped,
            v=torch.ones_like(x_reshaped),
            create_graph=True
        )
        return jacobian_reshape.view(batch_size, 1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.activation(x) @ self.weight) + self.bias

    def greek(self, x: torch.Tensor, prev_greek: torch.Tensor) -> torch.Tensor:
        """
        For a given forward pass with input `x`, i.e. `y = forward(x)`; we can compute the greek of the output.
        """
        batch_size = x.shape[0]
        return (prev_greek @ self.weight.T) * self.activation_derivative(x)
