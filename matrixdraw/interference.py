# from io import BytesIO
# from dataclasses import dataclass, field
# from typing import Literal, Any
# import ast
# import base64

# import numpy as np
# from numpy.typing import NDArray
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.figure import Figure

# # from draw import filter_kwargs


# def filter_kwargs(cls: type, **kwargs: Any) -> dict[str, Any]:
#     """Filter kwargs to only include those that are valid for the given class."""
#     return {k: v for k, v in kwargs.items() if k in {f.name for f in fields(cls)}}


# def truncate_colormap(cmap: str, minval: float=0.0, maxval:float=1.0, n:int=100):
#     colormap = plt.get_cmap(cmap)
#     new_cmap = mcolors.LinearSegmentedColormap.from_list(
#         'trunc({n},{a:.2f},{b:.2f})'.format(n=colormap.name, a=minval, b=maxval),
#         colormap(np.linspace(minval, maxval, n)))  # type: ignore
#     return new_cmap


# @dataclass
# class MatrixPlot:
#     """Visualize a 2D matrix"""
#     init_m: NDArray[Any]
#     size: float = 1
#     square: bool = True
#     fixed_height: bool = False
#     fixed_width: bool = False
#     max_w_and_h: tuple[int, int] | None = None  # (max width, max height) in pixels
#     pixel_scale: float = 0.001
#     vector_factor: float | None = 0.1  # display vector as a fraction of the square size
#     transpose_vectors: bool = False
#     raster: bool = True
#     quick: bool = False
#     downsample_factor: int = 1
#     downsample_kernel: Literal['mean', 'median', 'max_abs'] = 'max_abs'
#     color_norm: Literal['linear', 'log'] = 'linear'
#     cmap: str = 'RdBu'
#     cmap_truncate: float = 0.3
#     clip: float | int | None = 1
#     m: NDArray[Any] = field(default_factory=lambda: np.asarray([[0, 0], [0, 0]]), init=False)


#     def __post_init__(self) -> None:
#         self.ensure_2D()
#         if self.downsample_factor > 1:
#             self.downsample()
#         if self.transpose_vectors and 1 in self.m.shape:
#             self.m = self.m.transpose(0, 1)
#         self.m = np.flip(self.m, 0)  # flip to match the image coordinate system
        
#     def ensure_2D(self) -> None:
#         """Ensure that m is 2D"""
#         m = self.init_m.astype(np.float32)
#         while len(m.shape) < 2:
#             m = np.expand_dims(m, -1)
#         if m.ndim != 2:
#             raise ValueError("Matrix must have 2 or fewer dimensions!")
#         self.m = m
        

#     def kernel(self, block: NDArray) -> np.floating[Any]:  # type: ignore
#         match self.downsample_kernel:
#             case "mean":
#                 return np.mean(block)  # type: ignore
#             case "median":
#                 return np.median(block)  # type: ignore
#             case "max_abs":
#                 sup: np.floating[Any] = np.max(block)  # type: ignore
#                 inf: np.floating[Any] = np.min(block)  # type: ignore
#                 return sup if abs(sup) > abs(inf) else inf  # type: ignore


#     def downsample(self) -> None:
#         """Downsample a 2D matrix by factor k"""
#         h, w = self.m.shape
#         k = self.downsample_factor
#         new_h = (h + k - 1) // k
#         new_w = (w + k - 1) // k
#         result = np.zeros((new_h, new_w), dtype=np.float32)
#         for i in range(new_h):
#             for j in range(new_w):
#                 block = self.m[i * k : min((i + 1) * k, h), j * k : min((j + 1) * k, w)]
#                 result[i, j] = self.kernel(block)  # type: ignore
#         self.m = result


#     def get_figsize(self) -> tuple[float, float]:
#         """Get the figure size in inches"""
#         h, w = self.m.shape
#         s = self.size
#         if self.raster:
#             s *= 2  # account for Image retina option in display()
#         if self.max_w_and_h:
#             max_w, max_h = self.max_w_and_h
#             # if the matrix is wider than the max aspect ratio, scale by width
#             # if the matrix is taller than the max aspect ratio, scale by height
#             scaling_factor = min(max_w / w, max_h / h)
#             s = scaling_factor * s
#             # self.pixel_scale = s / self.size
#         if self.square:
#             width, height = s, s
#             if self.vector_factor and h == 1:  # is row vector
#                 height *= self.vector_factor
#             if self.vector_factor and w == 1:  # is column vector
#                 width *= self.vector_factor
#             return width, height
#         elif self.fixed_height:
#             return s * w / h, s
#         elif self.fixed_width:
#             return s, s * h / w
#         else:
#             pixel_size = s * self.pixel_scale
#             return pixel_size * w, pixel_size * h


#     def get_color_norm(self) -> mcolors.Normalize:
#         """Get the color normalization for the matrix"""
#         c = self.clip
#         if not c:
#             c = np.max(np.abs(self.m)).item()
#         if self.color_norm == "linear":
#             return mcolors.Normalize(vmin=-c, vmax=c)
#         elif self.color_norm == "log":
#             return mcolors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-c, vmax=c)
#         else:
#             raise ValueError(f"Unknown color norm: {self.color_norm}")


#     def get_matrix_figure(self) -> Figure:
#         """Create a plt figure of a 2D matrix"""
#         h, w = self.m.shape
#         fig, ax = plt.subplots()   # type: ignore[reportUnknownMemberType]
#         fig.set_size_inches(self.get_figsize())
#         fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#         norm = self.get_color_norm()
#         Xx, Yy = np.meshgrid(np.arange(w + 1), np.arange(h + 1), indexing="xy")  # type: ignore[reportUnknownMemberType]
#         X = Xx.astype(np.float32)  # for compatibility with matplotlib
#         Y = Yy.astype(np.float32)
#         cmap = truncate_colormap(self.cmap, self.cmap_truncate, 1-self.cmap_truncate)
#         pcm = ax.pcolormesh(X, Y, self.m, cmap=cmap, norm=norm, edgecolors="face")  # type: ignore[reportUnknownMemberType]
#         if not self.raster:  # 2x pcolormesh to prevent SVG seams:
#             pcm = ax.pcolormesh(X, Y, self.m, cmap=cmap, norm=norm, edgecolors="face")  # type: ignore[reportUnknownMemberType]
#         pcm.set_edgecolor("none")  # removes the edgecolors='face' distortion
#         ax.set_anchor("NW")
#         ax.axis("off")
#         return fig


#     def get_buffer(self) -> BytesIO:
#         """Get a buffer with the figure"""
#         fig = self.get_matrix_figure()
#         buffer = BytesIO()
#         format = "png" if self.raster else "svg"
#         plt.savefig(buffer, format=format, transparent=True, bbox_inches="tight")  # type: ignore[reportUnknownMemberType]
#         buffer.seek(0)
#         plt.close(fig)
#         return buffer


#     def save(self, filename: str) -> None:
#         fig = self.get_matrix_figure()  # type: ignore[reportUnknownMemberType]
#         format = "png" if self.raster else "svg"
#         plt.savefig(filename, format=format, transparent=True, bbox_inches="tight")  # type: ignore[reportUnknownMemberType]


# def get_matrix_grid_html(matrices: list[NDArray], **kwargs: Any) -> str:  # type: ignore
#     "For displaying matrices in a flexible grid, using IPython.display.HTML"
#     gap = kwargs.get("gap", 5)
#     css = f"<style>.matrix-container {{display: flex; flex-wrap: wrap; gap: {gap}px;}}</style>"
#     matrices_html: list[str] = []
#     for m in matrices:  # type: ignore
#         filtered_kwargs = filter_kwargs(MatrixPlot, **kwargs)
#         mplot = MatrixPlot(m, **filtered_kwargs)  # type: ignore
#         buffer_val = mplot.get_buffer().getvalue()
#         if mplot.raster:
#             img_str = base64.b64encode(buffer_val).decode('utf-8')
#             html_str = f'<img src="data:image/png;base64,{img_str}" />'
#         else:
#             html_str = buffer_val.decode('utf-8')
#         matrices_html.append(html_str)
#     html = f"{css}<div class='matrix-container'>{''.join(matrices_html)}</div>"
#     return html


# def plot(matrix: NDArray, **kwargs: Any) -> str:  # type: ignore
#     """Visualize matrices of 2 or fewer dimensions in an HTML grid"""
#     matrices = [matrix]  # type: ignore
#     html_str = get_matrix_grid_html(matrices, **kwargs)
#     return html_str




# def str_to_matrix(matrix_data_str: str) -> list[list[int | float]]:
#     matrix = ast.literal_eval(matrix_data_str)
#     assert len(matrix) > 0, "Matrix must have at least one row, got 0"
#     assert len(matrix[0]) > 0, "Matrix must have at least one column, got 0"
#     row_lengths = [len(row) for row in matrix]
#     assert all(all(isinstance(el, (int, float)) for el in row) for row in matrix), "Matrix must be a 2D Python list"
#     assert len(set(row_lengths))==1, f"All rows must have the same length, got: {row_lengths}"
#     return matrix


# def main(matrix_data_str: str, width: int, height: int) -> str:
#     output = ""
#     try:
#         max_w_and_h = (width, height)
#         matrix = str_to_matrix(matrix_data_str)
#         array = np.array(matrix)
#         output = plot(array, raster=False, clip=None, square=False, max_w_and_h=max_w_and_h, pixel_scale=0.0105, cmap='RdBu', cmap_truncate=0.1)
#     except Exception as e:
#         output = f"Error: {e}"
#     return output


# def generate_interference_pattern_numpy():
#     n_sines = 5
#     # Generate random coefficients for sine waves
#     coefficients = np.random.uniform(0, 10, (n_sines, 2))
    
#     # Create meshgrid
#     x = np.arange(0, n_sines, 0.1)
#     y = np.arange(0, n_sines, 0.1)
#     X, Y = np.meshgrid(x, y)
    
#     # Calculate sum of sine waves
#     result = np.zeros_like(X)
#     for a, b in coefficients:
#         result += np.sin(a * X + b * Y)
    
#     # Convert to list[list[float]]
#     result = np.round(result, decimals=2, out=None)
#     return result.tolist()

# def example_1() -> str:
#     """Larger example matrix for testing"""
#     matrix = generate_interference_pattern_numpy()
#     return str(matrix)

# def example_2() -> str:
#     """Example matrix for testing"""
#     matrix = '[[1, 2, 3], [4, -5, 6], [7, 8, 9]]'
#     return matrix


# # Example usage
# if __name__ == "__main__":
#     matrix_data_str = example_2()
#     max_w_and_h = (20, 20)
#     matrix = str_to_matrix(matrix_data_str)
#     array = np.array(matrix)
#     mplot = MatrixPlot(array, raster=True, clip=None, square=False, max_w_and_h=max_w_and_h, pixel_scale=0.0105, cmap='RdBu', cmap_truncate=0.1)  # type: ignore
#     mplot.save("test.png")
#     # print(example_2())