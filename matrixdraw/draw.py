from io import BytesIO
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from matrixdraw.color import Color


@dataclass
class PlotConfig:
    """Configuration for the matrix plot"""
    size: float = 1
    max_shape: tuple[int, int] | None = None  # (max width, max height) in pixels
    min_shape: tuple[int, int] | None = None  # (min width, min height) in pixels
    hardcoded_shape: tuple[int, int] | None = None  # (width, height) in pixels
    square: bool = False  # forces square aspect ratio
    pixel_size: float = 0.01  # pixel width in inches
    raster: bool = False
    color: Color = field(default_factory=Color)


@dataclass
class Matrix:
    """Visualize a 2D matrix"""
    m: NDArray[Any]
    conf: PlotConfig = field(default_factory=PlotConfig)

    @property
    def figsize(self) -> tuple[float, float]:
        """Get the figure size in inches"""
        c = self.conf
        if c.hardcoded_shape:
            return c.hardcoded_shape

        # Set initial size
        h, w = self.m.shape
        s: float = c.size

        # Scale to fit into width/height limits
        if c.max_shape:
            max_w, max_h = c.max_shape
            if w > max_w or h > max_h:
                s *= min(max_w/w, max_h/h)
        if c.min_shape:
            min_w, min_h = c.min_shape
            if w < min_w or h < min_h:
                s *= max(min_w/w, min_h/h)

        # Set final size
        s *= c.pixel_size
        if c.square:
            return s, s
        else:
            return s*w, s*h


    @property
    def figure(self) -> Figure:
        """Create a plt figure of a 2D matrix"""
        # Create a figure
        fig, ax = plt.subplots()   # type: ignore[reportUnknownMemberType]
        fig.set_size_inches(self.figsize)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Create a coordinate system
        m = np.flip(self.m, 0)  # flip to match the matplotlib image coordinate system
        h, w = m.shape
        x, y = np.meshgrid(np.arange(w + 1), np.arange(h + 1), indexing="xy")  # type: ignore[reportUnknownMemberType]
        x, y = x.astype(np.float32), y.astype(np.float32)  # for compatibility with matplotlib

        # Get color normalization and colormap
        col = self.conf.color
        cnorm = col.cnorm(np.max(np.abs(m)).item())
        cmap = col.cmap
    
        # Draw the matrix
        pcm = ax.pcolormesh(x, y, m, cmap=cmap, norm=cnorm, edgecolors="face")  # type: ignore[reportUnknownMemberType]
        if not self.conf.raster:  # 2x pcolormesh to prevent SVG seams:
            pcm = ax.pcolormesh(x, y, m, cmap=cmap, norm=cnorm, edgecolors="face")  # type: ignore[reportUnknownMemberType]

        # Figure formatting
        pcm.set_edgecolor("none")  # removes the edgecolors='face' distortion
        ax.set_anchor("NW")
        ax.axis("off")
        return fig


    def save(self, destination: str | BytesIO) -> None:
        fig = self.figure  # type: ignore[reportUnknownMemberType]
        format = "png" if self.conf.raster else "svg"
        plt.savefig(destination, format=format, transparent=True, bbox_inches="tight")  # type: ignore[reportUnknownMemberType]
        plt.close(fig)
