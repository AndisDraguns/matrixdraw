from typing import Any

import numpy as np
from numpy.typing import NDArray

from matrixdraw.draw import Matrix, PlotConfig, Color


def interference_pattern(n_sines: int = 5, size: int = 50) -> NDArray[Any]:
    # Generate random coefficients for sine waves
    coefficients = np.random.uniform(0, 10, (n_sines, 2))
    
    # Create meshgrid
    x = np.arange(0, n_sines, n_sines/size)
    y = np.arange(0, n_sines, n_sines/size)
    X, Y = np.meshgrid(x, y)
    
    # Sum the sine waves
    result = np.zeros_like(X)
    for a, b in coefficients:
        result += np.sin(a * X + b * Y)
    return np.round(result, decimals=2, out=None)


def draw_interference_pattern():
    array = interference_pattern()
    config = PlotConfig(color=Color('RdBu', 0.1))
    mplot = Matrix(array, config)
    mplot.save("pattern.svg")


# Example usage
if __name__ == "__main__":
    draw_interference_pattern()
