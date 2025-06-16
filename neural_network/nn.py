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
    def calculate_gradient(self, layer: int, node: int):
        """
        Given two indices that correspond to a node in a trainable, non-input layer, 
        calculate the partial derivative of it with respect to all of the weights and biases
        of the previous layer.
        """
        if layer == 0 or layer >= len(self.layers):
            raise IndexError("Invalid layer index. Must be greater than 0 or less than len(self.layers)")
        
        # Reset the adjoints of the weights and biases of the previous layer
        
    def zero_grad(self, layer: int):
        """
        Resets all of the adjoints of the weights and biases of a linear layer.
        """
        assert isinstance(self.layers[layer], Linear)
        linear_layer: Linear = self.layers[layer] # type: ignore

        for i in range(linear_layer.weights.shape[0]):
            for j in range(linear_layer.weights.shape[1]):
                linear_layer.weights[i][j].adjoint = 0.0

nn = NN([
    Linear(3, 5, bias=True),
    ReLU(),
    Linear(5, 10, bias=True),
    Sigmoid()
])

x = np.array([DiffScalar(1.0, 2.0), DiffScalar(-2.0, -3.0), DiffScalar(3.0, 0.1)]).T

x = nn.forward(x)

# nn.zero_grad(0)

print(nn.layers[0])

# GOAL: Calculate the derivative between a single node and all of the neurons in the layer RIGHT BEFORE IT