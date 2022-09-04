from typing import Tuple

import torch


class DmlModule(torch.nn.Module):

    def forward_with_greek(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.forward(x)
        # noinspection PyTypeChecker
        return y, torch.autograd.functional.jacobian(self.forward, x)

    @classmethod
    def convert(cls, module: torch.nn.Module) -> None:
        # TODO. Check that this is not a subclass of DmlModule already

        class DmlVersionOfModule(cls, type(module)):
            pass
        DmlVersionOfModule.__name__ = 'Dml' + module.__class__.__name__

        module.__class__ = DmlVersionOfModule
