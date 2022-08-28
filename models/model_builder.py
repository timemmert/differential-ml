from keras import Input
import tensorflow as tf
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam

from layers.backprop_dense import BackpropDense
from layers.backprop_dense_output import BackpropDenseOutput
from layers.forward_dense import ForwardDense

_LOSS_METRIC = mse


class ModelFactory:
    def __init__(
        self,
        input_dimension,
        output_dimension,
        lambda_j,
        n_layers,
        hidden_layer_sizes,
        optimizer=Adam(),
        lambda_=0,
        regularization_weight=0,
    ):
        self._n_layers = n_layers
        self._hidden_units = hidden_layer_sizes
        self._lambda = lambda_
        self.gamma = regularization_weight
        self._optimizer = optimizer

        self._input_dim = input_dimension
        self._lambda_j = lambda_j
        self._feed_forward_output_dim = output_dimension

    def build_model(self) -> Model:
        zs, zs_bar = self._build_layers()

        label_layer_y, label_layer_y_bar = self._setup_dummy_labels()
        model = self._build_keras_model(label_layer_y, label_layer_y_bar, zs, zs_bar)
        loss_function = self._build_loss_function(
            label_layer_y, label_layer_y_bar, zs, zs_bar, model
        )
        model.add_loss(loss_function)
        model.add_metric(mse(label_layer_y, zs[-1]), "metric")
        model.compile(
            optimizer=self._optimizer,
        )
        return model

    def _setup_dummy_labels(self):
        """
        These dummy labels are a small hack to make cost functions handle multiple network outputs.
        """
        label_layer_y = Input((self._feed_forward_output_dim,), name="dummy layer y")
        label_layer_y_bar = Input(
            (
                self._feed_forward_output_dim,
                self._input_dim,
            ),
            name="dummy layer y_bar",
        )
        return label_layer_y, label_layer_y_bar

    def _build_layers(self):
        L = self._n_layers
        zs, layers = self._build_feed_forward_layer(L)
        zs_bar = self._build_backprop_feed_forward_layer(L, layers, zs)
        return zs, zs_bar

    def _build_keras_model(self, label_layer_y, label_layer_y_bar, zs, zs_bar) -> Model:
        model = Model(
            inputs=[
                zs[0],
                # Those are dummies needed to construct the cost function
                label_layer_y,
                label_layer_y_bar,
            ],
            outputs=[zs[-1], zs_bar[-1]],
        )
        return model

    def _build_loss_function(self, label_layer_y, label_layer_y_bar, zs, zs_bar, model):
        alpha = 1.0 / (1.0 + self._lambda * self._input_dim)
        beta = 1.0 - alpha
        loss_function = (
            alpha * _LOSS_METRIC(zs[-1], label_layer_y)
            + beta
            * tf.reduce_mean(
                _LOSS_METRIC(
                    zs_bar[-1] * self._lambda_j, label_layer_y_bar * self._lambda_j
                ),
                axis=1,
            )
            + self.gamma * self._build_regularization_term(model)
        )

        return loss_function

    def _build_regularization_term(self, model):
        regularization = 0
        for layer in model.layers:
            if isinstance(layer, ForwardDense):
                weights = layer.weights[0]
                regularization += tf.norm(weights)
        return regularization

    def _build_feed_forward_layer(self, L):
        z0 = Input(self._input_dim, name="x = z0")
        zs = [z0]
        layers = []
        for l in range(1, L):
            activation = tf.nn.softplus
            if l == 1:
                activation = None
            layer = ForwardDense(
                units=self._hidden_units[l - 1],
                activation=activation,
                name=f"z{l}",
            )
            layers.append(layer)
            zs.append(layer(zs[l - 1]))
        layer_y = ForwardDense(
            units=self._feed_forward_output_dim,
            activation=tf.nn.softplus,
            name="y",
        )
        layers.append(layer_y)
        y = layer_y(zs[-1])
        zs.append(y)

        return zs, layers

    def _build_backprop_feed_forward_layer(self, L, layers, zs):
        zL_bar = tf.linalg.diag(tf.ones_like(zs[-1]), name=f"y_bar = z{L}_bar")
        zs_bar = [zL_bar]
        layers_bar = []
        for l in reversed(range(1, L)):
            layer_z_bar = BackpropDense(dense_layer=layers[l], name=f"z{l}_bar")
            layers_bar.append(layer_z_bar)
            z_bar = layer_z_bar(
                (
                    zs_bar[L - l - 1],
                    zs[l],
                )
            )
            zs_bar.append(z_bar)
        layer_y_bar = BackpropDenseOutput(dense_layer=layers[0], name="y_bar")
        layers_bar.append(layer_y_bar)
        y_bar = layer_y_bar(zs_bar[-1])
        zs_bar.append(y_bar)

        return zs_bar
