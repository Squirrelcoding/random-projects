import numpy as np

np.random.seed(42)


class Layer():
    def __init__(self) -> None:
        pass

    def forward(self, weights):
        pass

class Linear():
    """
    A neural network layer of dimension `n` with the weights initialized to random values.
    """
    def __init__(self, n: int) -> None:
        self.activations = np.array([np.random.rand() for _ in range(n)])
        self.dim = n

class Sigmoid():
    def __init__(self) -> None:
        pass

class NeuralNetwork():
    def __init__(self, layers: list[Linear]) -> None:
        self.layers = layers
        self.activation_functions = []
        self.weights = []
        self.biases = []

    def add_layer(self, layer: Linear, activation_function):
        self.layers.append(layer)
        self.activation_functions.append(activation_function)

        n = self.layers[-2].dim 
        m = layer.dim 

        # If the previous layer has N nodes and the new layer has M nodes then the weight matrix is N x M
        self.weights.append(np.random.rand(n, m))
        self.biases.append(np.random.rand(m))

