"""
TensorFlow functional helper utilities.
"""

from typing import Any, Tuple

import tensorflow as tf  # type: ignore


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
