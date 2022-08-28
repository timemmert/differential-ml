import tensorflow as tf
from keras.engine.base_layer import Layer
from keras.layers import Dense


class BackpropDense(Layer):
    def __init__(self, dense_layer: Dense, name=None):
        super(BackpropDense, self).__init__(name=name)
        self._dense_layer = dense_layer
        self.w = dense_layer.weights[0]

    def call(self, inputs, *args, **kwargs):
        z_bar_previous, z_mirror = inputs

        return (z_bar_previous @ tf.transpose(self.w)) * tf.nn.sigmoid(
            tf.expand_dims(z_mirror, axis=1)
        )
