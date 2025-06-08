from __future__ import annotations

import numpy as np

# Forward differentiation with some basic variables
class FVariable:
    def __init__(self, primal: float, tangent: float) -> None:
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
    def __repr__(self):
        return f"primal: {self.primal}, tangent: {self.tangent}"

negative_one = FVariable(-1.0, 0)

sigmoid = FVariable(1.0, 0) / (FVariable(1.0, 0.0) + FVariable(-4.0, 0).exp())

print(sigmoid)