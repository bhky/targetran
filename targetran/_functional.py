"""
Functional helper utilities.
"""

from typing import (
    Any, Callable, Iterable, List, Optional, Tuple, TypeVar
)

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import cv2  # type: ignore


T = TypeVar("T", np.ndarray, tf.Tensor)


def _map_single(
        fn: Callable[..., Tuple[T, T]],
        image_list: List[T],
        bboxes_list: List[T],
        iterable_args: Optional[List[Iterable[Any]]],
        *args: Any,
        **kwargs: Any
) -> Tuple[List[T], List[T]]:
    """
    Map each image and bboxes array/tensor from the lists, to the fn which
    takes as input a single image and bboxes, together with other arguments.
    Set iterable_args to None if not available.
    """
    iters = [image_list, bboxes_list, *iterable_args] if iterable_args \
        else [image_list, bboxes_list]
    pairs = [fn(*iterables, *args, **kwargs) for iterables in zip(*iters)]
    image_list, bboxes_list = zip(*pairs)
    return image_list, bboxes_list


# Numpy.

def _np_convert(x: Any) -> np.ndarray:
    return np.array(x, dtype=np.float32)


def _np_ragged_to_list(bboxes_ragged: np.ndarray) -> List[np.ndarray]:
    return [
        np.reshape(np.array(bboxes), (-1, 4)) for bboxes in bboxes_ragged
    ]


def _np_list_to_ragged(bboxes_list: List[np.ndarray]) -> np.ndarray:
    return np.array(
        [np.reshape(bboxes, (-1, 4)) for bboxes in bboxes_list], dtype=object
    )


def _np_stack_bboxes(bboxes_ragged: np.ndarray) -> np.ndarray:
    bboxes_list = _np_ragged_to_list(bboxes_ragged)
    all_bboxes = np.concatenate(bboxes_list, 0)
    assert np.shape(all_bboxes)[-1] == 4
    return all_bboxes


def _np_round_to_int(x: np.ndarray) -> np.ndarray:
    return np.rint(x).astype(dtype=np.int32)


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
    return _np_list_to_ragged(bboxes_list)


# TF.

def _tf_convert(x: Any) -> tf.Tensor:
    return tf.convert_to_tensor(np.array(x), dtype=tf.float32)


def _tf_ragged_to_list(bboxes_ragged: tf.Tensor) -> List[tf.Tensor]:
    return [
        tf.reshape(bboxes, (-1, 4)) for bboxes in bboxes_ragged.to_list()
    ]


def _tf_list_to_ragged(bboxes_list: List[tf.Tensor]) -> tf.Tensor:
    return tf.ragged.stack(bboxes_list)


def _tf_stack_bboxes(bboxes_ragged: tf.Tensor) -> tf.Tensor:
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    all_bboxes = tf.concat(bboxes_list, 0)
    assert np.shape(all_bboxes)[-1] == 4
    return all_bboxes


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
        bboxes_ragged: tf.Tensor,
) -> tf.Tensor:
    bboxes_nums = [len(bboxes) for bboxes in bboxes_ragged.to_list()]
    return tf.RaggedTensor.from_row_lengths(all_bboxes, bboxes_nums)
