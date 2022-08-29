import torch
from torch import nn


class DmlLinear(nn.Module):

    def __init__(self, input_dim: torch.Tensor, output_dim: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

        self.activation = nn.functional.softplus
        self.activation_derivative = nn.functional.sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation((x @ self.weight) + self.bias)

    def greek(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        For a given forward pass with input `x`, i.e. `y = forward(x)`; we can compute the greek of the output.
        """
        return (y @ self.weight.T) * self.activation_derivative(x)[:, None]

