"""
Functional helper utilities.
"""

from typing import Any, Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import cv2  # type: ignore


# Numpy.

def _np_convert(x: Any) -> np.ndarray:
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


# TF.

def _tf_convert(x: Any) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.float32)
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _tf_round_to_int(x: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.math.rint(x), dtype=tf.int32)


def _tf_pad_image(
        image: tf.Tensor,
        pad_offsets: tf.Tensor,
) -> tf.Tensor:
    """
    pad_offsets: [top, bottom, left, right]
    """
    height = int(tf.shape(image)[0])
    width = int(tf.shape(image)[1])
    target_height = int(pad_offsets[0]) + height + int(pad_offsets[1])
    target_width = int(pad_offsets[2]) + width + int(pad_offsets[3])
    return tf.image.pad_to_bounding_box(
        image,
        int(pad_offsets[0]), int(pad_offsets[2]),
        target_height, target_width
    )


def _tf_resize_image(
        image: tf.Tensor,
        dest_size: Tuple[int, int]
) -> tf.Tensor:
    """
    dest_size: (height, width)
    """
    return tf.image.resize(
        image, size=dest_size, method=tf.image.ResizeMethod.AREA
    )


def _tf_gather_image(image: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """
    indices: [[row_0, row_1, ...], [col_0, col_1, ...]]
    """
    return tf.gather_nd(image, tf.transpose(indices))
