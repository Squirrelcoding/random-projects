import numpy.typing as npt

def mse(input: npt.NDArray, target: npt.NDArray):
    assert input.shape == target.shape
    return (input - target) ** 2. / input.shape[0]
