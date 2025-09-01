from typing import Literal, Any

import numpy as np
from numpy.typing import NDArray


def ensure_2D(m: NDArray[Any]) -> NDArray[Any]:
    """Ensure that m is 2D"""
    # m = m.astype(np.float32)  # for compatibility with matplotlib
    while len(m.shape) < 2:
        m = np.expand_dims(m, -1)
    if m.ndim != 2:
        raise ValueError("Matrix must have 2 or fewer dimensions!")
    return m


def kernel(block: NDArray, kernel_type: Literal['mean', 'median', 'max_abs']) -> np.floating[Any]:  # type: ignore
    match kernel_type:
        case "mean":
            return np.mean(block)  # type: ignore
        case "median":
            return np.median(block)  # type: ignore
        case "max_abs":
            sup: np.floating[Any] = np.max(block)  # type: ignore
            inf: np.floating[Any] = np.min(block)  # type: ignore
            return sup if abs(sup) > abs(inf) else inf  # type: ignore


def downsample(m: NDArray[Any], k: int, kernel_type: Literal['mean', 'median', 'max_abs'] = 'max_abs') -> NDArray[Any]:
    """Downsample a 2D matrix by factor k"""
    h, w = m.shape
    new_h = (h + k - 1) // k
    new_w = (w + k - 1) // k
    result = np.zeros((new_h, new_w), dtype=np.float32)
    for i in range(new_h):
        for j in range(new_w):
            block = m[i * k : min((i + 1) * k, h), j * k : min((j + 1) * k, w)]
            result[i, j] = kernel(block, kernel_type)  # type: ignore
    return result
