"""
Functional helper utilities.
"""

from typing import Callable, Tuple

import numpy as np  # type: ignore
import scipy.ndimage  # type: ignore


def _np_array_map(
        fn: Callable[[np.ndarray], np.ndarray],
        iterable: np.ndarray
) -> np.ndarray:
    return np.array([fn(x) for x in iterable])


def _np_concat_map(
        fn: Callable[[np.ndarray], np.ndarray],
        iterable: np.ndarray
) -> np.ndarray:
    return np.concatenate([fn(x) for x in iterable])


def _np_multiply(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.multiply(x, y)


def _np_logical_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.logical_and(x, y)


def _np_resize(images: np.ndarray, dest_size: Tuple[int, int]) -> np.ndarray:
    """
    dest_size: (height, width)
    """
    h_factor = dest_size[0] / images.shape[1]
    w_factor = dest_size[1] / images.shape[2]
    return scipy.ndimage.zoom(images, (1.0, h_factor, w_factor, 1.0))


def _np_boolean_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    mask: boolean array
    """
    return x[mask]
