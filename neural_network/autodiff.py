from __future__ import annotations

import numpy as np

np.random.seed(42)

# Taken from https://huggingface.co/blog/andmholm/what-is-automatic-differentiation but expanded upon
class DiffScalar:
    def __init__(self, primal: float, adjoint: float = 0.0):
        self.primal = primal
        self.adjoint = adjoint

    def backward(self, adjoint):
        self.adjoint += adjoint

    def __add__(self, other: DiffScalar):
        # print(f"[DEBUG] Adding {self} and {other}")
        variable = DiffScalar(self.primal + other.primal)

        def backward(adjoint):
            # Let Z = x1 + x2

            # Accumulate gradient from output: df/dZ += incoming gradient
            variable.adjoint += adjoint

            # These are dZ/dx1 and dZ/dx2
            self_adjoint = adjoint * 1.0
            other_adjoint = adjoint * 1.0
            
            # Update df/dx1 and df/dx2 for each input variable
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __sub__(self, other: DiffScalar):
        # print(f"[DEBUG] Subtracting {self} and {other}")
        variable = DiffScalar(self.primal - other.primal)

        def backward(adjoint):
            # Let Z = x1 - x2

            # Accumulate gradient: add adjoint to df/dZ
            variable.adjoint += adjoint
            
            # Compute dZ/dx1 and dZ/dx2
            self_adjoint = adjoint * 1.0
            other_adjoint = adjoint * -1.0

            # Update dF/dx1 and dF/dx2
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __mul__(self, other: DiffScalar):
        # print(f"[DEBUG] Multiplying {self} and {other}")
        variable = DiffScalar(self.primal * other.primal)

        def backward(adjoint):
            # Let Z = x1 * x2
            # adjoint = dF/dZ

            # Update dF/dZ
            variable.adjoint += adjoint

            # Chain rule:
            # dZ/dx1 = x2
            # dZ/dx2 = x1
            # So:
            # dF/dx1 = dF/dZ * dZ/dx1 = adjoint * x2
            # dF/dx2 = dF/dZ * dZ/dx2 = adjoint * x1
            self_adjoint = adjoint * other.primal
            other_adjoint = adjoint * self.primal

            # Update dF/dx1 and dF/dx2
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def __truediv__(self, other: DiffScalar):
        # print(f"[DEBUG] Dividing {self} and {other}")
        variable = DiffScalar(self.primal / other.primal)

        def backward(adjoint):
            # Let Z = x1 / x2
            # adjoint = dF/dZ
            variable.adjoint += adjoint

            # Chain rule:
            # dZ/dx1 = 1 / x2
            # dZ/dx2 = - x1 / x2^2
            # dF/dx1 = df/dZ * dZ/dx1 = adjoint * 1/x2
            # dF/dx2 = df/dZ * dZ/dx2 = adjoint * -x1 / x2^2
            self_adjoint = adjoint * (1.0 / other.primal)
            other_adjoint = adjoint * (-1.0 * self.primal / other.primal**2)

            # Update dF/dx1 and dF / dx2
            self.backward(self_adjoint)
            other.backward(other_adjoint)

        variable.backward = backward
        return variable

    def exp(self):
        variable = DiffScalar(np.exp(self.primal))

        def backward(adjoint):
            # Let Z = exp(x1)
            # adjoint = dF/dZ
            variable.adjoint += adjoint

            # Chain rule:
            # dF/dx1 = dF/dZ * dZ/dx1 = adjoint * exp(x1)
            self_adjoint = adjoint * np.exp(self.primal)

            # Update dF/dx1
            self.backward(self_adjoint)

        variable.backward = backward
        return variable

    def __repr__(self) -> str:
        return f"DiffScalar({self.primal}, {self.adjoint})"
    def __str__(self) -> str:
        return f"DiffScalar({self.primal}, {self.adjoint})"
        
    def __lt__(self, other: DiffScalar) -> bool:
        return self.primal < other.primal



def sigmoid(input: DiffScalar):
    return DiffScalar(1.0) / (DiffScalar(1.0) + (DiffScalar(-1.0) * input).exp())

a0 = DiffScalar(np.random.random())
a1 = DiffScalar(np.random.random())

b2 = DiffScalar(np.random.random())
b3 = DiffScalar(np.random.random())
b4 = DiffScalar(np.random.random())
b5 = DiffScalar(np.random.random())
b6 = DiffScalar(np.random.random())

# a2
w21 = DiffScalar(np.random.random())
w22 = DiffScalar(np.random.random())

# a3
w31 = DiffScalar(np.random.random())
w32 = DiffScalar(np.random.random())


# a4
w41 = DiffScalar(np.random.random())
w42 = DiffScalar(np.random.random())

# a5
w51 = DiffScalar(np.random.random())
w52 = DiffScalar(np.random.random())
w53 = DiffScalar(np.random.random())

# a6
w61 = DiffScalar(np.random.random())
w62 = DiffScalar(np.random.random())
w63 = DiffScalar(np.random.random())

a2 = w21 * a0 + w22 * a1 + b2
a3 = w31 * a0 + w32 * a1 + b3
a4 = w41 * a0 + w42 * a1 + b4

a5 = sigmoid(w51 * a2 + w52 * a3 + w53 * a4 + b5)
a6 = sigmoid(w61 * a2 + w62 * a3 + w63 * a4 + b6)

a6.backward(1.0)
# for var in [a0, a1, a2, a3, a4, a5, a6,
#             w21, w22, w31, w32, w41, w42, w51, w52, w53, w61, w62, w63,
#             b2, b3, b4, b5, b6]:
#     var.adjoint = 0.0

a5.backward(1.0)
# print(a2, a3, a4)

# print(a0)
# print(a1)
# print(a6)