from typing import Literal
from dataclasses import dataclass

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


@dataclass
class Color:
    colormap: str = 'RdBu'  # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    truncate: float = 0.3  # selects a subset of colormap colors
    clip: float | int | None = None  # if abs(x) < clip, color(x) = color(clip * sign(x))
    scaling: Literal['linear', 'log'] = 'linear'  # color(x) vs color(log(x))
    cmap_steps: int = 256  # number of steps in the colormap

    def cnorm(self, backup_clip: float) -> mcolors.Normalize:
        """Get the color normalization for the matrix"""
        c = self.clip or backup_clip
        if self.scaling == "linear":
            return mcolors.Normalize(vmin=-c, vmax=c)
        elif self.scaling == "log":
            return mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-c, vmax=c)
        else:
            raise ValueError(f"Unknown color normalization type: {self.scaling}")

    @property
    def cmap(self) -> mcolors.Colormap | mcolors.LinearSegmentedColormap:
        colormap = plt.get_cmap(self.colormap)
        a, b = self.truncate, 1 - self.truncate  # new colormap range
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            f'trunc({colormap.name},{a:.2f},{b:.2f})',
            colormap(np.linspace(a, b, self.cmap_steps)))
        return new_cmap
