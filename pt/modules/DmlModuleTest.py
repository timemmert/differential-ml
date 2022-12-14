import numpy as np
import pytest
import torch

from pt.modules.DmlLinear import DmlLinear
from pt.modules.DmlModule import DmlModule, AlreadyDmlModuleException


def test_conversion():
    test_tensor_input = np.array([1.0, 2.0, 3.0])

    class MyNet(torch.nn.Module):

        def forward(self, x):
            return x ** 2

    my_net = MyNet()
    assert np.any(my_net(torch.as_tensor(test_tensor_input)).numpy() == test_tensor_input ** 2)

    DmlModule.convert(my_net)
    assert my_net.__class__.__name__ == 'DmlMyNet'

    y, greek = my_net.forward_with_greek(torch.as_tensor(test_tensor_input))
    assert np.any(torch.diagonal(greek).numpy() == test_tensor_input * 2)
    assert np.any(y.numpy() == test_tensor_input ** 2)
    assert np.any(my_net(torch.as_tensor(test_tensor_input)).numpy() == y.numpy())


def test_subclass_conversion():
    dml_linear_net = DmlLinear(1, 1, torch.nn.Identity())

    with pytest.raises(AlreadyDmlModuleException):
        DmlModule.convert(dml_linear_net)

