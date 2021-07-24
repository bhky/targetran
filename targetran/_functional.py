"""
Functional helper utilities.
"""

from typing import Any, Callable, Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import cv2  # type: ignore


# Numpy.

def _np_map_idx_fn(
        fn: Callable[[int], Tuple[np.ndarray, np.ndarray]],
        batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The fn here should take an idx as the only input.
    """
    tuples = [fn(idx) for idx in range(batch_size)]
    images_seq, bboxes_seq = list(zip(*tuples))
    return np.array(images_seq), np.array(bboxes_seq)


def _np_to_single_fn(
        fn: Callable[..., Tuple[np.ndarray, np.ndarray]]
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

    def single_fn(
            image: np.ndarray,
            bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        new_images, new_bboxes_ragged = fn(
            np.array([image]), np.array([bboxes])
        )
        return new_images[0], new_bboxes_ragged[0]

    return single_fn


def _np_convert(x: Any) -> np.ndarray:
    return np.array(x, dtype=np.float32)


def _np_range(start: int, end: int, step: int) -> np.ndarray:
    return np.arange(start, end, step)


def _np_stack_bboxes(bboxes_ragged: np.ndarray) -> np.ndarray:
    bboxes_list = [
        np.reshape(np.array(bboxes), (-1, 4)) for bboxes in bboxes_ragged
    ]
    all_bboxes = np.concatenate(bboxes_list, 0)
    assert np.shape(all_bboxes)[-1] == 4
    return all_bboxes


def _np_round_to_int(x: np.ndarray) -> np.ndarray:
    return np.rint(x.astype(dtype=np.float32)).astype(dtype=np.int32)


def _np_logical_and(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.logical_and(x, y)


def _np_pad_images(
        images: np.ndarray,
        pad_offsets: np.ndarray
) -> np.ndarray:
    """
    pad_offsets: [top, bottom, left, right]
    """
    pad_width = (  # From axis 0 to 3.
        (0, 0),
        (int(pad_offsets[0]), int(pad_offsets[1])),
        (int(pad_offsets[2]), int(pad_offsets[3])),
        (0, 0)
    )
    return np.pad(images, pad_width=pad_width, constant_values=0)


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


def _np_make_bboxes_ragged(
        all_bboxes: np.ndarray,
        bboxes_ragged: np.ndarray,
) -> np.ndarray:
    bboxes_nums = [len(bboxes) for bboxes in bboxes_ragged]
    indices = np.cumsum(bboxes_nums)[:-1]
    bboxes_list = np.split(all_bboxes, indices, 0)
    return np.array(
        [np.reshape(bboxes, (-1, 4)) for bboxes in bboxes_list], dtype=object
    )


# TF.

def _tf_map_idx_fn(
        fn: Callable[[tf.Tensor], Tuple[tf.Tensor, tf.RaggedTensor]],
        batch_size: int
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    """
    The fn here should take an idx tensor as the only input.
    """
    return tf.map_fn(  # type: ignore
        fn, tf.range(batch_size),
        fn_output_signature=(
            tf.TensorSpec(None, tf.float32),
            tf.RaggedTensorSpec((None, 4), tf.float32, ragged_rank=1)
        )
    )


def _tf_to_single_fn(
        fn: Callable[..., Tuple[tf.Tensor, tf.RaggedTensor]]
) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:

    def single_fn(
            image: tf.Tensor,
            bboxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        new_images, new_bboxes_ragged = fn(
            tf.expand_dims(image, 0), tf.RaggedTensor.from_tensor([bboxes])
        )
        return new_images[0], new_bboxes_ragged[0]

    return single_fn


def _tf_convert(x: Any) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.float32)
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _tf_stack_bboxes(bboxes_ragged: tf.RaggedTensor) -> tf.Tensor:
    bboxes_ragged = tf.cast(bboxes_ragged, dtype=tf.float32)
    return tf.reshape(
        bboxes_ragged.to_tensor(default_value=np.nan), (-1, 4)
    )


def _tf_round_to_int(x: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.math.rint(x), dtype=tf.int32)


def _tf_pad_images(
        images: tf.Tensor,
        pad_offsets: tf.Tensor,
) -> tf.Tensor:
    """
    pad_offsets: [top, bottom, left, right]
    """
    height = int(tf.shape(images)[1])
    width = int(tf.shape(images)[2])
    target_height = int(pad_offsets[0]) + height + int(pad_offsets[1])
    target_width = int(pad_offsets[2]) + width + int(pad_offsets[3])
    return tf.image.pad_to_bounding_box(
        images,
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


def _tf_make_bboxes_ragged(
        all_bboxes: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
) -> tf.RaggedTensor:
    row_lengths = bboxes_ragged.row_lengths()
    batch_size = len(row_lengths)
    return tf.RaggedTensor.from_tensor(
        tf.reshape(all_bboxes, (batch_size, -1, 4)),
        lengths=row_lengths
    )
