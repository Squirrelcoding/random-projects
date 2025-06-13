import numpy as np
import numpy.typing as npt

from autodiff import DiffScalar

np.random.seed(42)

def dot(a: npt.NDArray, b: npt.NDArray):
    assert a.shape[-1] == b.shape[0]
    res = np.array([[DiffScalar(0.0) for w in range(a.shape[0])] for j in range(b.shape[-1])])
    for ai, bi in zip(a, b):
        res += ai * bi
    return res


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
        # Wrap each float in a DiffScalar
        self.weights = np.array([[DiffScalar(w) for w in row] 
                                 for row in np.random.rand(out_features, in_features)], dtype=object)

        if bias:
            self.bias = np.array([DiffScalar(b) 
                                  for b in np.random.rand(out_features)], dtype=object)
        else:
            self.bias = np.array([DiffScalar(0.0)
                                  for _ in range(out_features)], dtype=object)

    def forward(self, input_weights: np.ndarray):
        # Assumes input_weights is a vector of DiffScalars (shape: [in_features])
        print(f"[DEBUG] {self.weights}, {input_weights}")
        return dot(self.weights, input_weights) + self.bias


# Activation functions

class Sigmoid(Layer):
    def __init__(self) -> None:
        pass
    def forward(self, input_weights: npt.NDArray):
        return DiffScalar(1.0) / (DiffScalar(1.0) + np.exp(-input_weights))

class ReLU(Layer):
    def __init__(self) -> None:
        pass
    def forward(self, input_weights: npt.NDArray):
        return input_weights * (input_weights > DiffScalar(0.0))
