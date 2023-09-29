from dataclasses import dataclass

import torch

from pt.modules.DmlFeedForward import DmlFeedForward
from pt.modules.DmlLoss import DmlLoss

from differential_ml.pt.modules.device import global_device


@dataclass
class DmlTrainingStep:
    x: torch.Tensor
    y_out: torch.Tensor
    y_target: torch.Tensor
    greek_out: torch.Tensor
    greek_target: torch.Tensor
    loss: torch.Tensor
    net: DmlFeedForward


@dataclass
class DmlTrainer:
    net: DmlFeedForward
    loss: DmlLoss
    optimizer: torch.optim.Optimizer

    def step(self, x: torch.Tensor, y_target: torch.Tensor, greek_target: torch.Tensor) -> DmlTrainingStep:
        y_out, greek_out = self.net.forward_with_greek(x)
        loss = self.loss(y_out, y_target, greek_out, greek_target, self.net)
        self.net.zero_grad()
        loss.backward()

        def closure():
            self.optimizer.zero_grad()  # Zero out previous gradients
            y_out_, greek_out_ = self.net.forward_with_greek(x)
            loss_ = self.loss(y_out_, y_target, greek_out_, greek_target, self.net)
            loss_.backward()
            return loss_

        self.optimizer.step(closure)

        # This is not necessary, this is just for callbacks and visualization of training.
        return DmlTrainingStep(x, y_out, y_target, greek_out, greek_target, loss, self.net)
