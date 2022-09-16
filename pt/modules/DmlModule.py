import warnings
from typing import Tuple

import torch


class AlreadyDmlModuleException(Exception):
    pass


class DmlModule(torch.nn.Module):

    def forward_with_greek(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.forward(x)
        # noinspection PyTypeChecker
        return y, torch.autograd.functional.jacobian(self.forward, x)

    @classmethod
    def convert(cls, module: torch.nn.Module) -> None:

        warnings.warn('This feature is experimental.')

        cls.check_model_is_not_dml_module(module)

        class DmlVersionOfModule(cls, type(module)):
            pass

        DmlVersionOfModule.__name__ = 'Dml' + module.__class__.__name__

        module.__class__ = DmlVersionOfModule

    @classmethod
    def check_model_is_not_dml_module(cls, module: torch.nn.Module) -> None:
        if issubclass(module.__class__, cls):
            raise AlreadyDmlModuleException(
                'Conversion of modules with existing DML logic to DML modules is not possible.'
                )
