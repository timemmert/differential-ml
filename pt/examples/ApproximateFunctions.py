import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from pt.DmlTrainer import DmlTrainer
from pt.modules.DmlDataset import DmlDataset
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
    n_train = 100
    n_test = 1000
    x_train, y_train, dydx_train = polynomial_and_trigonometric(n_train)
    x_test, y_test, dydx_test = polynomial_and_trigonometric(n_test)

    """if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        torch.set_default_device(mps_device)
        torch.set_default_dtype(torch.float32)
        print(x)
    else:
        print("MPS device not found.")"""

    normalizer = DataNormalizer()

    normalizer.initialize_with_data(x_raw=x_train, y_raw=y_train, dydx_raw=dydx_train)
    x_train_normalized, y_train_normalized, dy_dx_train_normalized = normalizer.normalize_all(
        x_train,
        y_train,
        dydx_train,
    )

    n_layers = 2
    hidden_layer_sizes = 100
    lr_dml = 0.1
    lr_std = 0.1
    batch_size = 32
    n_epochs = 100
    activation = torch.nn.Softmax(-1)
    # activation = torch.nn.ReLU()

    dml_net = DmlFeedForward(
        normalizer.input_dimension,
        normalizer.output_dimension,
        n_layers,
        hidden_layer_sizes,
        activation,
    )

    dml_loss = DmlLoss(
        _lambda=1,  # Weight of differentials in Loss
        _input_dim=normalizer.input_dimension,
        _lambda_j=normalizer.lambda_j,
    )
    dml_sgd = torch.optim.Adam(lr=lr_dml, params=dml_net.parameters())
    dml_trainer = DmlTrainer(dml_net, dml_loss, optimizer=dml_sgd)

    # Same as the DML net, but will be trained with lambda = 0
    net = DmlFeedForward(
        normalizer.input_dimension,
        normalizer.output_dimension,
        n_layers,
        hidden_layer_sizes,
        activation,
    )
    loss = DmlLoss(
        _lambda=0,
        _input_dim=normalizer.input_dimension,
        _lambda_j=normalizer.lambda_j,
    )
    sgd = torch.optim.Adam(lr=lr_std, params=net.parameters())
    trainer = DmlTrainer(net, loss, optimizer=sgd)

    dml_dataset = DmlDataset(x_train_normalized, y_train_normalized, dy_dx_train_normalized)

    train_size = int(0.8 * len(dml_dataset))
    valid_size = len(dml_dataset) - train_size
    train_set, validation_set = torch.utils.data.random_split(dml_dataset, [train_size, valid_size])
    shuffle = True  # Set to True if you want to shuffle the data

    # Create a DataLoader using the custom dataset
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    dataloader_valid = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle)

    for epoch in range(n_epochs):
        pbar_train = tqdm(dataloader_train)
        pbar_valid = tqdm(dataloader_valid, disable=True)
        # Train loop
        for batch in pbar_train:
            inputs = batch['x']  # Access input features
            targets = batch['y']  # Access target labels
            gradients = batch['dydx']  # Access dy/dx values
            step_dml = dml_trainer.step(
                inputs,
                targets,
                gradients,
            )
            step = trainer.step(
                inputs,
                targets,
                gradients,
            )
            pbar_train.set_description(f"Loss DML: {step_dml.loss.item()}, Loss: {step.loss.item()}")
        with torch.no_grad():
            valid_error_dml = 0
            valid_error = 0
            for batch in pbar_valid:
                inputs = batch['x']  # Access input features
                targets = batch['y']  # Access target labels

                outputs_dml = dml_net(inputs)
                outputs = net(inputs)

                valid_error_dml += float(MSELoss()(outputs_dml, targets))
                valid_error += float(MSELoss()(outputs, targets))
            print(f"Validation Loss DML {valid_error_dml}\nValidation Loss {valid_error}")

    dml_net.eval()
    y_out_test = [
        normalizer.unscale_y(dml_net(torch.as_tensor(x, dtype=torch.float32)).item()) for x in normalizer.scale_x(x_test)
    ]
    print('Test Error DML:', np.mean((np.array(y_out_test) - y_test) ** 2))
    plt.scatter(x_test, y_out_test, s=.1)
    plt.scatter(x_test, y_test, s=.1)
    plt.show()

    net.eval()
    y_out_test = [
        normalizer.unscale_y(net(torch.as_tensor(x, dtype=torch.float32)).item()) for x in
        normalizer.scale_x(x_test)
    ]
    print('Test Error Standard:', np.mean((np.array(y_out_test) - y_test) ** 2))



