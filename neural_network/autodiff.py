# Forward 

class Variable:
    def __init__(self, primal, tangent) -> None:
        self.primal = primal
        self.tangent = tangent
    def __add__(self, other):
        res = Variable(self.primal + other.primal, self.tangent + other.tangent)
        return res
    def __mul__(self, other):
        res = Variable(
            self.primal * other.primal, 
            self.tangent * other.primal + other.tangent * self.primal
        )
        return res
    def __div__(self, other):
        res = Variable(
            self.primal / other.primal, 
            (self.tangent * other.primal - self.primal * other.tangent) / (other.primal)**2
        )
        return res

x1 = Variable(4.0, 1.0)
x2 = Variable(5.0, 0.0)