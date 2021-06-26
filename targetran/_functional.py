"""
Functional helper utilities.
"""

from typing import Any, List, Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import scipy.ndimage  # type: ignore


def _np_convert(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _np_multiply(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.multiply(x, y)


def _np_logical_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.logical_and(x, y)


def _np_resize_image(
        image: np.ndarray,
        dest_size: Tuple[int, int]
) -> np.ndarray:
    """
    dest_size: (height, width)
    """
    h_factor = dest_size[0] / image.shape[0]
    w_factor = dest_size[1] / image.shape[1]
    return scipy.ndimage.zoom(image, (h_factor, w_factor, 1.0))


def _np_boolean_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    mask: boolean array
    """
    return x[mask]


def _np_make_bboxes_list(
        all_bboxes: np.ndarray,
        bboxes_nums: List[int]
) -> List[np.ndarray]:
    indices = np.cumsum(bboxes_nums)[:-1]
    bboxes_list = np.split(all_bboxes, indices, 0)
    return [np.reshape(bboxes, (-1, 4)) for bboxes in bboxes_list]


def _tf_convert(x: Any) -> tf.Tensor:
    return tf.convert_to_tensor(np.array(x), dtype=tf.float32)


def _tf_resize_image(
        image: tf.Tensor,
        dest_size: Tuple[int, int]
) -> tf.Tensor:
    """
    dest_size: (height, width)
    """
    return tf.image.resize(image, size=dest_size)


def _tf_make_bboxes_list(
        all_bboxes: tf.Tensor,
        bboxes_nums: List[int]
) -> List[tf.Tensor]:
    bboxes_list = tf.split(all_bboxes, bboxes_nums, 0)
    return [tf.reshape(bboxes, (-1, 4)) for bboxes in bboxes_list]
