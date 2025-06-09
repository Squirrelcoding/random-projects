import numpy as np
import numpy.typing as npt

from autodiff import DiffScalar

np.random.seed(42)

class Layer():
    """
    Base class for layers.
    """
    def forward(self, input_weights: npt.NDArray):
        return input_weights

class Linear(Layer):
    """
    A linear layer
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.weights = np.random.rand(out_features, in_features).astype(dtype=DiffScalar)
        if bias:
            self.bias = np.random.rand(out_features).T
        else:
            self.bias = np.zeros(out_features)

    def forward(self, input_weights: npt.NDArray):
        return np.matmul(self.weights, input_weights) + self.bias

# Activation functions

class Sigmoid(Layer):
    def __init__(self) -> None:
        pass
    def forward(self, input_weights: npt.NDArray):
        return 1.0 / (1 + np.exp(-input_weights))

class ReLU(Layer):
    def __init__(self) -> None:
        pass
    def forward(self, input_weights: npt.NDArray):
        return input_weights * (input_weights > 0)
