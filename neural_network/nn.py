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

nn = NN([
    Linear(3, 5, bias=True),
    ReLU(),
    Linear(5, 10, bias=True),
    Sigmoid()
])

x = np.array([1.0, -2.0, 3.0])

print(nn.forward(x))
