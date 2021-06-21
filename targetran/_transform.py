"""
Image and target transform utilities.
"""

from typing import Callable, List, Tuple, TypeVar

import numpy as np
import tensorflow as tf

T = TypeVar("T", np.ndarray, tf.Tensor)


def _make_output_bboxes_list(
        bboxes_nums: List[int],
        all_bboxes: np.ndarray,
        split_func: Callable[[T, T, int], List[T]],
        reshape_func: Callable[[T, Tuple[int, int]], T]
) -> List[T]:
    """
    Helper function for making the output list of bboxes.
    """
    indices = np.cumsum(bboxes_nums)[:-1]
    bboxes_list = split_func(all_bboxes, indices, 0)  # Along axis 0.
    return [reshape_func(bboxes, (-1, 4)) for bboxes in bboxes_list]


def _flip_left_right(
        images: T,
        bboxes_list: List[T],
        shape_func: Callable[[T], Tuple[int, ...]],
        concat_func: Callable[[List[T], int], T],
        split_func: Callable[[T, T, int], List[T]],
        reshape_func: Callable[[T, Tuple[int, int]], T]
) -> Tuple[T, List[T]]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    image_shape = shape_func(images)
    assert len(image_shape) == 4

    image_width = image_shape[2]

    images = images[..., ::-1, :]

    all_bboxes = concat_func(bboxes_list, 0)  # Along axis 0.
    assert shape_func(all_bboxes)[-1] == 4

    all_bboxes[:, :1] = image_width - all_bboxes[:, :1] - all_bboxes[:, 2:3]

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    bboxes_list = _make_output_bboxes_list(
        bboxes_nums, all_bboxes, split_func, reshape_func
    )

    return images, bboxes_list


def _flip_up_down(
        images: T,
        bboxes_list: List[T],
        shape_func: Callable[[T], Tuple[int, ...]],
        concat_func: Callable[[List[T], int], T],
        split_func: Callable[[T, T, int], List[T]],
        reshape_func: Callable[[T, Tuple[int, int]], T]
) -> Tuple[T, List[T]]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    image_shape = shape_func(images)
    assert len(image_shape) == 4

    image_height = image_shape[1]

    images = images[:, ::-1, ...]

    all_bboxes = concat_func(bboxes_list, 0)  # Along axis 0.
    assert shape_func(all_bboxes)[-1] == 4

    all_bboxes[:, 1:2] = image_height - all_bboxes[:, 1:2] - all_bboxes[:, 3:]

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    bboxes_list = _make_output_bboxes_list(
        bboxes_nums, all_bboxes, split_func, reshape_func
    )

    return images, bboxes_list


def _rotate_90(
        images: np.ndarray,
        bboxes_list: List[np.ndarray],
        shape_func: Callable[[T], Tuple[int, ...]],
        transpose_func: Callable[[T, Tuple[int, ...]], T],
        concat_func: Callable[[List[T], int], T],
        split_func: Callable[[T, T, int], List[T]],
        reshape_func: Callable[[T, Tuple[int, int]], T]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Rotate 90 degrees anti-clockwise.
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    image_shape = shape_func(images)
    assert len(image_shape) == 4

    image_width = image_shape[2]

    images = transpose_func(images, (0, 2, 1, 3))[:, ::-1, :, :]

    all_bboxes = concat_func(bboxes_list, 0)  # Along axis 0.
    assert shape_func(all_bboxes)[-1] == 4

    all_bboxes = concat_func([
        all_bboxes[:, 1:2],
        image_width - all_bboxes[:, :1] - all_bboxes[:, 2:3],
        all_bboxes[:, 3:],
        all_bboxes[:, 2:3],
    ], 1)  # Along axis 1.

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    bboxes_list = _make_output_bboxes_list(
        bboxes_nums, all_bboxes, split_func, reshape_func
    )

    return images, bboxes_list
