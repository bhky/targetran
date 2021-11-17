"""
Numpy functional helper utilities.
"""

from typing import Any, Tuple

import cv2  # type: ignore
import numpy as np

from targetran._typing import NPArray


def _np_convert(x: Any) -> NPArray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return np.array(x, dtype=np.float32)


def _np_range(start: int, end: int, step: int) -> NPArray:
    return np.arange(start, end, step)  # type: ignore


def _np_cast_to_int(x: NPArray) -> NPArray:
    return x.astype(dtype=np.int32)


def _np_round_to_int(x: NPArray) -> NPArray:
    return np.rint(x.astype(dtype=np.float32)).astype(dtype=np.int32)  # type: ignore


def _np_logical_and(x: NPArray, y: NPArray) -> NPArray:
    return np.logical_and(x, y)  # type: ignore


def _np_pad_image(
        image: NPArray,
        pad_offsets: NPArray
) -> NPArray:
    """
    pad_offsets: [top, bottom, left, right]
    """
    pad_width = (  # From axis 0 to 2.
        (int(pad_offsets[0]), int(pad_offsets[1])),
        (int(pad_offsets[2]), int(pad_offsets[3])),
        (0, 0)
    )
    return np.pad(image, pad_width=pad_width, constant_values=0)  # type: ignore


def _np_resize_image(
        image: NPArray,
        dest_size: Tuple[int, int]
) -> NPArray:
    """
    dest_size: (image_height, image_width)
    """
    resized_image: NPArray = cv2.resize(  # pylint: disable=no-member
        image,
        dsize=(dest_size[1], dest_size[0]),
        interpolation=cv2.INTER_AREA  # pylint: disable=no-member
    )
    return resized_image


def _np_boolean_mask(x: NPArray, mask: NPArray) -> NPArray:
    """
    mask: boolean array
    """
    return x[mask]  # type: ignore


def _np_gather_image(image: NPArray, indices: NPArray) -> NPArray:
    """
    indices: [[row_0, row_1, ...], [col_0, col_1, ...]]
    """
    return image[tuple(indices)]  # type: ignore
