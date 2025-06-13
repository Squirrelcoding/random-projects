from layer import *

import numpy.typing as npt

class NN():
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input_weights: npt.NDArray):
        x = input_weights
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def calculate_gradient(self, input: int, output: int):
        """
        Given two indices that correspond to two distinct trainable (`Linear`) layers,
        calculate the gradients of each activation with respect to the weights of the 
        previous layer.
        """
        pass

nn = NN([
    Linear(3, 5, bias=True),
    ReLU(),
    Linear(5, 10, bias=True),
    Sigmoid()
])

x = np.array([DiffScalar(1.0), DiffScalar(-2.0), DiffScalar(3.0)])

print(nn.forward(x))
