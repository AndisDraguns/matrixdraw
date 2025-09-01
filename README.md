# matrixdraw
Matrix visualization with seamless vector graphics

![](./matrixdraw/examples/pattern.svg)

Installation:
```bash
pip install .
```

Example:
```python
import numpy as np
from matrixdraw.draw import Matrix, PlotConfig

array = np.array([[1, 2, 3], [4, -5, 6], [7, 8, 9]])
config = PlotConfig(size=50)
Matrix(array, config).save("example.svg")
```

Live demo [here](www.draguns.me/visualizer.html)
