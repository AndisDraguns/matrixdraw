import numpy as np
from matrixdraw.draw import Matrix, PlotConfig

array = np.array([[1, 2, 3], [4, -5, 6], [7, 8, 9]])
config = PlotConfig(size=50)
Matrix(array, config).save("simple.svg")
