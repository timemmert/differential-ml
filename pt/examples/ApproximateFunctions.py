import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from pt.DmlTrainer import DmlTrainer
from pt.modules.DmlFeedForward import DmlFeedForward
from pt.modules.DmlLoss import DmlLoss
from util.data_util import DataNormalizer


def polynomial_and_trigonometric(n):
    # function y = 0.1*x³ + 0.2x² + 3*sin(3x)
    # derivative dydx = 3x² + 4x + cos(x)
    x = np.random.uniform(low=-10, high=10, size=n)
    y = 0.1 * (x ** 3) + 0.2 * (x ** 2) + 3 * np.sin(3 * x) + 10
    dydx = 0.3 * (x ** 2) + 0.4 * x + 9 * np.cos(3 * x)

    # plt.scatter(x, y)
    # plt.show()
    return x.reshape(-1, 1), y.reshape(-1, 1), dydx.reshape(-1, 1)


if __name__ == "__main__":

    n_train = 10000
    n_test = 1000
    x_train, y_train, dydx_train = polynomial_and_trigonometric(n_train)
    x_test, y_test, dydx_test = polynomial_and_trigonometric(n_test)

    normalizer = DataNormalizer()

    normalizer.initialize_with_data(x_raw=x_train, y_raw=y_train, dydx_raw=dydx_train)
    x_train_normalized, y_train_normalized, dy_dx_train_normalized = normalizer.normalize_all(
        x_train,
        y_train,
        dydx_train,
    )

    n_layers = 3
    hidden_layer_sizes = 40

    net = DmlFeedForward(
        normalizer.input_dimension,
        normalizer.output_dimension,
        n_layers,
        hidden_layer_sizes,
        torch.nn.Softmax(-1),
    )
    print((list(net.parameters())))

    loss = DmlLoss(
        _loss_metric=torch.nn.MSELoss(),
        _lambda=.5,
        _input_dim=normalizer.input_dimension,
        _lambda_j=normalizer.lambda_j,
    )
    sgd = torch.optim.Adam(net.parameters())
    trainer = DmlTrainer(net, loss, optimizer=sgd)

    pbar = tqdm(zip(x_train_normalized, y_train_normalized, dy_dx_train_normalized), total=n_train)
    for x, y, dydx in pbar:
        step = trainer.step(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(dydx, dtype=torch.float32),
        )
        pbar.set_postfix({'Loss': step.loss.item()})

    print((list(net.parameters())))

    net.eval()
    y_out_test = [
        normalizer.unscale_y(net(torch.as_tensor(x, dtype=torch.float32)).item()) for x in normalizer.scale_x(x_test)
    ]

    print('Error:', np.mean((np.array(y_out_test) - y_test) ** 2))

    plt.scatter(x_test, y_out_test, s=.1)
    plt.scatter(x_test, y_test, s=.1)
    plt.show()
