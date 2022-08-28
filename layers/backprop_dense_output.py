import tensorflow as tf
from keras.engine.base_layer import Layer
from keras.layers import Dense
from tensorflow import Tensor


class BackpropDenseOutput(Layer):
    def __init__(self, dense_layer: Dense, name=None):
        super(BackpropDenseOutput, self).__init__(name=name)
        self._dense_layer = dense_layer
        self.w = dense_layer.weights[0]

    def call(self, inputs: Tensor, *args, **kwargs):
        z_bar_previous = inputs
        return z_bar_previous @ tf.transpose(self.w)
