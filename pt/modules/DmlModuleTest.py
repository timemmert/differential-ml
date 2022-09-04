import numpy as np
import torch

from pt.modules.DmlModule import DmlModule


def test_conversion():
    test_tensor_input = np.array([1.0, 2.0, 3.0])
    class MyNet(torch.nn.Module):

        def forward(self, x):
            return x ** 2

    my_net = MyNet()
    assert np.any(my_net(torch.as_tensor(test_tensor_input)).numpy() == test_tensor_input ** 2)

    DmlModule.convert(my_net)
    assert np.any(my_net(torch.as_tensor(test_tensor_input)).numpy() == test_tensor_input ** 2)

    _, greek = my_net.forward_with_greek(torch.as_tensor(test_tensor_input))
    assert np.any(torch.diagonal(greek).numpy() == test_tensor_input * 2)
