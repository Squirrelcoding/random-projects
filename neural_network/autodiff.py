from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Forward differentiation with some basic variables
class FVariable:
    def __init__(self, primal: npt.NDArray, tangent: npt.NDArray) -> None:
        self.primal = primal
        self.tangent = tangent
    def __add__(self, other: FVariable) -> FVariable:
        return FVariable(
            self.primal + other.primal, 
            self.tangent + other.tangent
        )
    def __sub__(self, other: FVariable) -> FVariable:
        return FVariable(
            self.primal - other.primal, 
            self.tangent - other.tangent
        )
    def __mul__(self, other: FVariable) -> FVariable:
        return FVariable(
            self.primal * other.primal, 
            other.primal * self.tangent + self.primal * other.tangent
        )
    def __truediv__(self, other: FVariable) -> FVariable:
        return FVariable(
            self.primal / other.primal,
            (self.tangent * other.primal - self.primal * other.tangent) / (other.primal**2)
        )
    def exp(self) -> FVariable:
        exp_val = np.exp(self.primal)
        return FVariable(exp_val, self.tangent * exp_val)
    def sin(self) -> FVariable:
        return FVariable(np.sin(self.primal), self.tangent * np.cos(self.primal))
    def cos(self) -> FVariable:
        return FVariable(np.cos(self.primal), self.tangent * -np.sin(self.primal))
    def log(self) -> FVariable:
        return FVariable(np.log(self.primal), self.tangent / self.primal)
    def pow(self, exponent: float) -> FVariable:
        if exponent == -1.0:
            return FVariable(1.0 / self.primal, self.tangent / self.primal**2)
        return FVariable(
            np.float_power(self.primal, exponent), 
            self.tangent * exponent * np.float_power(self.primal, exponent - 1.0)
        )
    def __repr__(self):
        return f"primal: {self.primal}, tangent: {self.tangent}"

# Taken from https://huggingface.co/blog/andmholm/what-is-automatic-differentiation
class BVariable:
    def __init__(self, primal, adjoint=0.0):
        self.primal = primal
        self.adjoint = adjoint

    def backward(self, adjoint):
        self.adjoint += adjoint

    def __add__(self, other: BVariable):
        variable = BVariable(self.primal + other.primal)

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

    def __sub__(self, other: BVariable):
        variable = BVariable(self.primal - other.primal)

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

    def __mul__(self, other: BVariable):
        variable = BVariable(self.primal * other.primal)

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

    def __truediv__(self, other: BVariable):
        variable = BVariable(self.primal / other.primal)

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
    
    def sin(self):
        variable = BVariable(np.sin(self.primal))

        def backward(adjoint):
            # Let Z = sin(x1)
            # adjoint = dF/dZ
            variable.adjoint += adjoint

            # Chain rule:
            # dF/dx1 = dF/dZ * dZ/dx1 = adjoint * cos(x1)
            self_adjoint = adjoint * np.cos(self.primal)

            # Update dF/dx1
            self.backward(self_adjoint)

        variable.backward = backward
        return variable

    def __repr__(self) -> str:
        return f"primal: {self.primal}, adjoint: {self.adjoint}"
