import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential, Input, activations
from keras.layers import Dense
from keras.losses import mse
from keras.optimizers import Adam, Nadam
import tensorflow as tf
from tf.models.model_builder import ModelFactory
from util.data_util import DataNormalizer


def polynomial_and_trigonometric(n):
    # function y = 0.1*x³ + 0.2x² + 3*sin(3x)
    # derivative dydx = 3x² + 4x + cos(x)
    x = np.random.uniform(low=-10, high=10, size=n)
    y = 0.1 * (x ** 3) + 0.2 * (x ** 2) + 3 * np.sin(3 * x) + 10
    dydx = 0.3 * (x ** 2) + 0.4 * x + 9 * np.cos(3 * x)

    #plt.scatter(x, y)
    #plt.show()
    return x.reshape(-1, 1), y.reshape(-1, 1), dydx.reshape(-1, 1)

ep_train = 100
def train_differential_ml():
    global history
    model_factory = ModelFactory(
        input_dimension=normalizer.input_dimension,
        output_dimension=normalizer.output_dimension,
        lambda_j=normalizer.lambda_j,
        n_layers=n_layers,
        hidden_layer_sizes=hidden_layer_sizes,
        lambda_=.5,
    )
    model = model_factory.build_model()
    cut = int(n_train * 0.66)
    history = model.fit(
        [
            xTrainNormalized[:cut],
            yTrainNormalized[:cut],
            dy_dxTrainNormalized[:cut],
        ],
        validation_data=(
            [
                xTrainNormalized[cut:],
                yTrainNormalized[cut:],
                dy_dxTrainNormalized[cut:],
            ],
            None,
        ),
        batch_size=32,
        epochs=ep_train,
        verbose=True,
    )
    plt.plot(history.history["metric"])
    plt.show()
    return model

def _build_prediction_dummy_tensors(x, dml_model):
    shape_dummy_y = dml_model.output[0].shape.as_list()
    shape_dummy_y[0] = x.shape[0]
    shape_dummy_dy_dx = dml_model.output[1].shape.as_list()
    shape_dummy_dy_dx[0] = x.shape[0]
    dummy_tensor_y = np.zeros(shape_dummy_y)
    dummy_tensor_dydx = np.zeros(shape_dummy_dy_dx)
    return dummy_tensor_y, dummy_tensor_dydx


def train_vanilla():
    model_vanilla = Sequential(
        layers=[
            Input(shape=(normalizer.input_dimension,)),
            Dense(40, activation=activations.relu),
            Dense(40, activation=activations.relu),
            Dense(40, activation=activations.relu),
            Dense(normalizer.output_dimension)
        ]
    )
    model_vanilla.compile(optimizer=Adam(), loss=mse)
    cut = int(n_train*0.66)
    history = model_vanilla.fit(
        xTrainNormalized[:cut],
        yTrainNormalized[:cut],
        validation_data=(xTrainNormalized[cut:], yTrainNormalized[cut:],),
        epochs=ep_train,
        batch_size=128,
        verbose=True,
    )
    plt.plot(history.history["val_loss"], label="val")
    plt.plot(history.history["loss"], label="train")
    plt.legend()
    plt.show()
    return model_vanilla


if __name__ == "__main__":

    n_train = 10000
    n_test = 1000
    x_train, y_train, dydx_train = polynomial_and_trigonometric(n_train)
    x_test, y_test, dydx_test = polynomial_and_trigonometric(n_test)

    normalizer = DataNormalizer()

    normalizer.initialize_with_data(x_raw=x_train, y_raw=y_train, dydx_raw=dydx_train)
    (
        xTrainNormalized,
        yTrainNormalized,
        dy_dxTrainNormalized,
    ) = normalizer.normalize_all(x_train, y_train, dydx_train)

    n_layers = 3
    hidden_layer_sizes = [40] * n_layers

    model_dml = train_differential_ml()
    model_vanilla = train_vanilla()

    # Test
    dummy_tensor_y, dummy_tensor_dydx = _build_prediction_dummy_tensors(x_test, model_dml)
    x_scaled = normalizer.scale_x(x_test)

    y_pred_dml_scaled = model_dml.predict([x_scaled, dummy_tensor_y, dummy_tensor_dydx])[0]
    y_pred_dml = normalizer.unscale_y(y_pred_dml_scaled)

    y_pred_vanilla_scaled = model_vanilla.predict(x_scaled)
    y_pred_vanilla = normalizer.unscale_y(y_pred_vanilla_scaled)

    error_dml = tf.reduce_mean(mse(y_pred_dml, y_test))
    error_vanlla = tf.reduce_mean(mse(y_pred_vanilla, y_test))
    print(f"Vanilla error = {error_vanlla}")
    print(f"DML error = {error_dml}")
    s=.2
    plt.scatter(x_test, y_test, s=s)
    plt.scatter(x_test, y_pred_vanilla, s=s, label="vanilla")
    plt.scatter(x_test, y_pred_dml, s=s, label="dml")
    plt.legend()
    plt.show()
