from dataclasses import dataclass

import torch

from pt.modules.DmlFeedForward import DmlFeedForward
from pt.modules.DmlLoss import DmlLoss


@dataclass
class DmlTrainingStep:
    x: torch.Tensor
    y_out: torch.Tensor
    y_target: torch.Tensor
    greek_out: torch.Tensor
    greed_target: torch.Tensor
    loss: torch.Tensor
    net: DmlFeedForward


@dataclass
class DmlTrainer:
    net: DmlFeedForward
    loss: DmlLoss
    optimizer: torch.optim.Optimizer

    def step(self, x: torch.Tensor, y_target: torch.Tensor, greek_target: torch.Tensor) -> DmlTrainingStep:
        y_out, greek_out = self.net.forward_with_outputs(x)
        loss = self.loss(y_out, y_target, greek_out, greek_target, self.net)
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()

        # This is not necessary, this is just for callbacks and visualization of training.
        return DmlTrainingStep(x, y_out, y_target, greek_out, greek_target, loss, self.net)
