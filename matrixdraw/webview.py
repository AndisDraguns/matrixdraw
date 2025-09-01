from io import BytesIO
from dataclasses import fields
from typing import Any
import ast
import base64

import numpy as np
from numpy.typing import NDArray

from matrixdraw.draw import Matrix


def filter_kwargs(cls: type, **kwargs: Any) -> dict[str, Any]:
    """Filter kwargs to only include those that are valid for the given class."""
    return {k: v for k, v in kwargs.items() if k in {f.name for f in fields(cls)}}


def get_matrix_grid_html(matrices: list[NDArray], **kwargs: Any) -> str:  # type: ignore
    "For displaying matrices in a flexible grid, using IPython.display.HTML"
    gap = kwargs.get("gap", 5)
    css = f"<style>.matrix-container {{display: flex; flex-wrap: wrap; gap: {gap}px;}}</style>"
    matrices_html: list[str] = []
    for m in matrices:  # type: ignore
        filtered_kwargs = filter_kwargs(Matrix, **kwargs)
        mplot = Matrix(m, **filtered_kwargs)  # type: ignore
        buffer = BytesIO()
        mplot.save(buffer)
        buffer.seek(0)
        buffer_val = buffer.getvalue()
        if mplot.conf.raster:
            img_str = base64.b64encode(buffer_val).decode('utf-8')
            html_str = f'<img src="data:image/png;base64,{img_str}" />'
        else:
            html_str = buffer_val.decode('utf-8')
        matrices_html.append(html_str)
    html = f"{css}<div class='matrix-container'>{''.join(matrices_html)}</div>"
    return html


def str_to_matrix(matrix_data_str: str) -> list[list[int | float]]:
    matrix = ast.literal_eval(matrix_data_str)
    assert len(matrix) > 0, "Matrix must have at least one row, got 0"
    assert len(matrix[0]) > 0, "Matrix must have at least one column, got 0"
    row_lengths = [len(row) for row in matrix]
    assert all(all(isinstance(el, (int, float)) for el in row) for row in matrix), "Matrix must be a 2D Python list"
    assert len(set(row_lengths))==1, f"All rows must have the same length, got: {row_lengths}"
    return matrix


def main(matrix_data_str: str, width: int, height: int) -> str:
    output = ""
    try:
        max_w_and_h = (width, height)
        matrix = str_to_matrix(matrix_data_str)
        array = np.array(matrix)
        output = plot(array, raster=False, clip=None, square=False, max_w_and_h=max_w_and_h, pixel_scale=0.0105, cmap='RdBu', cmap_truncate=0.1)
    except Exception as e:
        output = f"Error: {e}"
    return output

def plot(matrix: NDArray, **kwargs: Any) -> str:  # type: ignore
    """Visualize matrices of 2 or fewer dimensions in an HTML grid"""
    matrices = [matrix]  # type: ignore
    html_str = get_matrix_grid_html(matrices, **kwargs)
    return html_str

# matrix = str_to_matrix('[[1, 2, 3], [4, -5, 6], [7, 8, 9]]')