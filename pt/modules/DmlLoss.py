from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import torch
from torch.nn import MSELoss

from pt.modules.DmlFeedForward import DmlFeedForward


@dataclass
class DmlLoss:
    _lambda: float
    _input_dim: int
    _lambda_j: float
    regularization_scale: float = 0.0

    def __call__(
        self,
        y_out: torch.Tensor,
        y_target: torch.Tensor,
        greek_out: torch.Tensor,
        greek_target: torch.Tensor,
        net: DmlFeedForward,
    ) -> torch.Tensor:
        ml_term = self.ml_loss_scale * self.ml_loss(y_out, y_target)
        dml_term = self.dml_loss_scale * self.dml_loss(greek_out, greek_target)
        regularization_term = 0.0 # self.regularization_scale * self.model_regularization(net)
        return ml_term + dml_term + regularization_term

    @property
    def _mse_loss(self):
        return MSELoss()

    def ml_loss(self, y_out: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        return self._mse_loss(y_out, y_target)

    @cached_property
    def ml_loss_scale(self) -> torch.Tensor:
        return torch.tensor(1.0 / (1.0 + self._lambda * self._input_dim))

    @property
    def _lambda_j_torch(self):
        return torch.tensor(self._lambda_j)

    def dml_loss(self, greek_out: torch.Tensor, greek_target: torch.Tensor) -> torch.Tensor:
        return torch.mean(self._mse_loss(greek_out * self._lambda_j_torch, greek_target * self._lambda_j_torch), dim=-1)

    @cached_property
    def dml_loss_scale(self) -> torch.Tensor:
        return 1.0 - self.ml_loss_scale

    @staticmethod
    def model_regularization(net: DmlFeedForward) -> torch.Tensor:
        return torch.sum(torch.as_tensor([torch.norm(layer.weight) for layer in net.layers_as_list]), dtype=torch.float32)
