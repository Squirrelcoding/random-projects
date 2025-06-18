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
        Resets all of the adjoints of activations of the layers.
        """
        for layer in self.layers:
            for activation in layer.activations:
                activation.adjoint = 0.0
            if isinstance(layer, Linear):
                for i in range(layer.weights.shape[0]):
                    for j in range(layer.weights.shape[1]):
                        layer.weights[i][j].adjoint = 0.0
                for bias in layer.bias:
                    bias.adjoint = 0.0

    def backward(self, d: npt.NDArray):
        """
        The backwards pass of the training loop. Calculates the derivative of the loss function with respect to 
        each layer's activations and updates the weights and biases (if applicable).
        """

        for layer in reversed(self.layers):
            d = layer.backward(d)
    def __repr__(self) -> str:
        return str(self.layers[0])

def mse_loss(pred: npt.NDArray, target: npt.NDArray) -> DiffScalar:
    """
    pred and target are both arrays of DiffScalars.
    Returns a single DiffScalar representing MSE loss.
    """
    assert pred.shape == target.shape
    n = pred.size
    total = DiffScalar(0.0)
    for i in range(n):
        diff = pred[i] - target[i]
        total += diff * diff
    return total * DiffScalar((1.0 / n), 0.0)

nn = NN([
    Linear(3, 5, bias=True),
    ReLU(),
    Linear(5, 1, bias=True),
    Sigmoid()
])

# Dummy data: 100 samples, 3 features each
np.random.seed(0)
X = np.random.randn(100, 3)

# Targets: sigmoid of sum of inputs (for simple regression-like task)
raw_Y = 1 / (1 + np.exp(-X.sum(axis=1)))
# Wrap targets in DiffScalar
Y = [np.array([DiffScalar(y)]) for y in raw_Y]

# Training parameters
epochs = 50

for epoch in range(epochs):
    total_loss = 0.0

    for x_sample, y_sample in zip(X, Y):
        # Wrap inputs as DiffScalars
        input_diff = np.array([DiffScalar(x) for x in x_sample])
        # Forward pass
        output = nn.forward(input_diff)
        # print(x_sample, y_sample, output)

        # Compute loss (both `output` and `y_sample` are DiffScalars)
        loss = mse_loss(output, y_sample)
        total_loss += loss.primal

        # Backward pass
        nn.zero_grad()
        loss.backward(1.0)
        d = np.array([out.adjoint for out in output])
        nn.backward(d)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X)}")


print(nn.forward(np.array([DiffScalar(0.2), DiffScalar(0.4), DiffScalar(0.3)])))