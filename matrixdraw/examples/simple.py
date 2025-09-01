import numpy as np
from matrixdraw.draw import Matrix


def draw_simple_matrix():
    mplot = Matrix(np.array([[1, 2, 3], [4, -5, 6], [7, 8, 9]]))
    mplot.save("simple.svg")


if __name__ == "__main__":
    draw_simple_matrix()
