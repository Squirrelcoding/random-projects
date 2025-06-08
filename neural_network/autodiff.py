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

# Seed for df/dy
vn1 = FVariable(3.0, 0.0)
v0 = FVariable(-4.0, 1.0)

v4 = (vn1 * v0).pow(-1.0)

print(v4)
