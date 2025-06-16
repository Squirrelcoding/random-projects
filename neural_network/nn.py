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
    def calculate_gradient(self, idx: int):
        """
        Given an index that corresponds to a layer, calculate the partial derivative of it with respect to all of the weights and biases of the previous layer.
        """
        if idx == 0 or idx >= len(self.layers):
            raise IndexError("Invalid layer index. Must be greater than 0 or less than len(self.layers)")
        
        # Loop through the nodes of the layer
        for node in self.layers[idx].activations:
            # Reset the adjoints of the weights and biases of the previous layer
            self.zero_grad(idx - 1)
            pass
        
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

nn.zero_grad(2)

print(nn.layers[0])

# GOAL: Calculate the derivative between a single node and all of the neurons in the layer RIGHT BEFORE IT
# The thing that is confusing me the most right now is how to actually go about how to calculate the gradient.
# Originally the goal was to simply call node.backward(1.0) but then I realized that in real neural networks
# like the one being built right now, we don't acutllay have independent variables that we can call .backward()
# on. Rather, for each layer we just have a matrix of weights and a bias vector. So what should we do if we want
# to calculate the gradients between two layers?
# Let's say that the previous layer has L inputs, M outputs and the current layer has M inputs and N outputs (nodes). So we have (L x M) and (M x N).
# ((L + 1) x M) matrix representing the weights and biases of the previous layer
# ((M + 1) x N) matrix representing the weights and biases of the current layer.

# There are N nodes, and for each node there are M weights and 1 bias. Thus, the gradient matrix should be like 
# an (M + 1) x N matrix. However, recall that in gradient descent we just take the average of all of the column
# entries to end up with an (M + 1) x 1 gradient vector that we use to adjust everything

# So for each of the N activations in the current layer, ask: how sensitive is this node with respect to all of
# the nodes in the previous layer?

# BUT HOW DO WE ACTUALLY DO THIS? 
# One way that we *could* possibly do it is to manually calculate the activation of the new node and then
# call the .backward(1.0) method on it. However, this feels like a waste of memory, when, in a sense, we already
# have the node with us.

# DOING THE BACKPROPAGATION ONLY MAKES SENSE WHEN WE ALREADY HAVE ACTIVATIONS.
# Okay, so all of the layers now have an activation field.
# 