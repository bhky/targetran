"""
Numpy functional helper utilities.
"""
from typing import Tuple

import cv2
import numpy as np

from targetran._typing import (
    ArrayLike,
    NDAnyArray,
    NDFloatArray,
    NDIntArray,
    NDBoolArray,
)
from targetran.utils import Interpolation

_INTERPOLATION_DICT = {
    Interpolation.NEAREST: cv2.INTER_NEAREST,  # pylint: disable=no-member
    Interpolation.BILINEAR: cv2.INTER_LINEAR,  # pylint: disable=no-member
    Interpolation.BICUBIC: cv2.INTER_CUBIC,  # pylint: disable=no-member
    Interpolation.AREA: cv2.INTER_AREA,  # pylint: disable=no-member
}


def _np_convert(x: ArrayLike) -> NDFloatArray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    return np.array(x, dtype=np.float32)


def _np_range(start: int, end: int, step: int) -> NDIntArray:
    return np.arange(start, end, step, dtype=np.int32)


def _np_cast_to_int(x: NDAnyArray) -> NDIntArray:
    return x.astype(dtype=np.int32)


def _np_round_to_int(x: NDAnyArray) -> NDIntArray:
    return np.rint(x.astype(dtype=np.float32)).astype(dtype=np.int32)


def _np_logical_and(x: NDBoolArray, y: NDBoolArray) -> NDBoolArray:
    return np.logical_and(x, y)


def _np_pad_image(
        image: NDFloatArray,
        pad_offsets: NDIntArray
) -> NDFloatArray:
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
        image: NDFloatArray,
        dest_size: Tuple[int, int],
        interpolation: Interpolation
) -> NDFloatArray:
    """
    dest_size: (image_height, image_width)
    """
    # pylint: disable=no-member
    resized_image: NDFloatArray = cv2.resize(
        image,
        dsize=(dest_size[1], dest_size[0]),
        interpolation=_INTERPOLATION_DICT[interpolation]
    )
    # pylint: enable=no-member
    return resized_image


def _np_boolean_mask(x: NDAnyArray, mask: NDBoolArray) -> NDAnyArray:
    """
    mask: boolean array
    """
    return x[mask]


def _np_gather_image(image: NDFloatArray, indices: NDIntArray) -> NDFloatArray:
    """
    indices: [[row_0, row_1, ...], [col_0, col_1, ...]]
    """
    return image[tuple(indices)]  # type: ignore
