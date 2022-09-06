from keras.layers import Dense


class ForwardDense(Dense):
    def call(self, inputs):
        return self.activation(inputs) @ self.weights[0] + self.weights[1]
