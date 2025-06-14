import numpy as np
import numpy.typing as npt

from autodiff import DiffScalar

np.random.seed(42)

def dot(A: np.ndarray, B: np.ndarray):
    assert A.shape[-1] == B.shape[0]
    B= B.reshape(-1, 1)
    result = np.empty((A.shape[0], B.shape[1]), dtype=object)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            acc = DiffScalar(0.0)
            for k in range(A.shape[1]):
                acc = acc + (A[i, k] * B[k, j])
            result[i, j] = acc
    return result.reshape(-1)

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
        r = dot(self.weights, input_weights) + self.bias
        return r

# Activation functions

def sigmoid_scalar(x: DiffScalar):
    return DiffScalar(1.0) / (DiffScalar(1.0) + x.exp())

def sigmoid(x: np.ndarray):
    sigmoid_vec = np.vectorize(sigmoid_scalar)
    return sigmoid_vec(x)

class Sigmoid(Layer):
    def __init__(self) -> None:
        pass
    def forward(self, input_weights: npt.NDArray):
        return sigmoid(input_weights)

def relu_scalar(x: DiffScalar):
    if x.primal > 0:
        out = DiffScalar(x.primal)
        def backward(adjoint):
            x.backward(adjoint * 1.0)
        out.backward = backward
    else:
        out = DiffScalar(0.0)
        def backward(adjoint):
            x.backward(0.0)
        out.backward = backward
    return out

def relu(x: np.ndarray):
    relu_vec = np.vectorize(relu_scalar)
    return relu_vec(x)

class ReLU(Layer):
    def __init__(self) -> None:
        pass
    def forward(self, input_weights: npt.NDArray):
        return relu(input_weights)

W = np.array([[DiffScalar(0.1), DiffScalar(0.2)],
              [DiffScalar(0.3), DiffScalar(0.4)]])
x = np.array([[DiffScalar(1.0)], [DiffScalar(2.0)]])

out = dot(W, x)

# print(out)