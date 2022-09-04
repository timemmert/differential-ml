from typing import Tuple

import torch


class DmlModule(torch.nn.Module):

    def forward_with_greek(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.forward(x)
        # noinspection PyTypeChecker
        return y, torch.autograd.functional.jacobian(self.forward, x)
