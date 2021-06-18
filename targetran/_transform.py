"""
Image and target transform utilities.
"""

from typing import Callable, List, Tuple, TypeVar

import numpy as np
import tensorflow as tf

T = TypeVar("T", np.ndarray, tf.Tensor)


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
    assert shape_func(images)[-1] == 3
    image_width = shape_func(images)[2]

    images = images[..., ::-1, :]

    all_bboxes = concat_func(bboxes_list, 0)  # Along axis 0.
    assert shape_func(all_bboxes)[-1] == 4

    all_bboxes[:, :1] = image_width - all_bboxes[:, :1] - all_bboxes[:, 2:3]

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    indices = np.cumsum(bboxes_nums)[:-1]
    bboxes_list = split_func(all_bboxes, indices, 0)  # Along axis 0.

    return images, [reshape_func(bboxes, (-1, 4)) for bboxes in bboxes_list]


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
    assert shape_func(images)[-1] == 3
    image_height = shape_func(images)[1]

    images = images[:, ::-1, ...]

    all_bboxes = concat_func(bboxes_list, 0)  # Along axis 0.
    assert shape_func(all_bboxes)[-1] == 4

    all_bboxes[:, 1:2] = image_height - all_bboxes[:, 1:2] - all_bboxes[:, 3:]

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    indices = np.cumsum(bboxes_nums)[:-1]
    bboxes_list = split_func(all_bboxes, indices, 0)  # Along axis 0.

    return images, [reshape_func(bboxes, (-1, 4)) for bboxes in bboxes_list]
