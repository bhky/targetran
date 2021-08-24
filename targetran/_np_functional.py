"""
Numpy functional helper utilities.
"""

from typing import Any, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore


def _np_convert(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return np.array(x, dtype=np.float32)


def _np_range(start: int, end: int, step: int) -> np.ndarray:
    return np.arange(start, end, step)


def _np_round_to_int(x: np.ndarray) -> np.ndarray:
    return np.rint(x.astype(dtype=np.float32)).astype(dtype=np.int32)


def _np_logical_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.logical_and(x, y)


def _np_pad_image(
        image: np.ndarray,
        pad_offsets: np.ndarray
) -> np.ndarray:
    """
    pad_offsets: [top, bottom, left, right]
    """
    pad_width = (  # From axis 0 to 2.
        (int(pad_offsets[0]), int(pad_offsets[1])),
        (int(pad_offsets[2]), int(pad_offsets[3])),
        (0, 0)
    )
    return np.pad(image, pad_width=pad_width, constant_values=0)


def _np_resize_image(
        image: np.ndarray,
        dest_size: Tuple[int, int]
) -> np.ndarray:
    """
    dest_size: (height, width)
    """
    return cv2.resize(
        image, dsize=(dest_size[1], dest_size[0]), interpolation=cv2.INTER_AREA
    )


def _np_boolean_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    mask: boolean array
    """
    return x[mask]


def _np_gather_image(image: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    indices: [[row_0, row_1, ...], [col_0, col_1, ...]]
    """
    return image[tuple(indices)]
