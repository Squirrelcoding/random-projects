from collections.abc import Callable
import numpy.typing as npt
from layer import *

class NN():
    def __init__(self, layers: list[Layer]) -> None:
        # loss_fn: Callable[[npt.NDArray, npt.NDArray], DiffScalar]
        self.layers = layers

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, input_weights: npt.NDArray):
        x = input_weights
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def zero_grad(self):
        """
        Resets all of the adjoints of the weights and biases of all the linear layers.
        """
        for layer in self.layers:
            for activation in layer.activations:
                activation.adjoint = 0.0
            if isinstance(layer, Linear):
                for i in range(layer.weights.shape[0]):
                    for j in range(layer.weights.shape[1]):
                        layer.weights[i][j].adjoint = 0.0

    def backward(self, loss: DiffScalar):
        """
        The backwards pass of the training loop. Calculates the derivative of the loss function with respect to 
        each layer's activations and updates the weights and biases (if applicable).
        """
    
        # NOT THIS. CALCULATE THE DERIVATIVE OF THE LOSS WITH RESPECT TO EACH OF THE FINAL ACTIVATIONS
        d: npt.NDArray = np.array([1.0 for _ in range(self.layers[-1].activations.shape[0])])
        for layer in reversed(self.layers):
            d = layer.backward(d)

def mse_loss(a: npt.NDArray, b: npt.NDArray):
    return DiffScalar(((a - b)**2).mean(), 0.0)

nn = NN([
    Linear(3, 5, bias=True),
    ReLU(),
    Linear(5, 10, bias=True),
    Sigmoid()
])

x = np.array([DiffScalar(1.0, 2.0), DiffScalar(-2.0, -3.0), DiffScalar(3.0, 0.1)]).T
