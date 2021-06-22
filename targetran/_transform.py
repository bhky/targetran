"""
Image and target transform utilities.
"""

from typing import Callable, List, Tuple, TypeVar

import numpy as np
import tensorflow as tf

T = TypeVar("T", np.ndarray, tf.Tensor, float)


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
    images_shape = shape_func(images)
    assert len(images_shape) == 4

    image_width = images_shape[2]

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
    images_shape = shape_func(images)
    assert len(images_shape) == 4

    image_height = images_shape[1]

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
    images_shape = shape_func(images)
    assert len(images_shape) == 4

    image_width = images_shape[2]

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


def _crop_and_resize(
        images: T,
        bboxes_list: List[T],
        offset_fractions: T,
        shape_func: Callable[[T], Tuple[int, ...]],
        multiply_func: Callable[[T, T], T],
        rint_func: Callable[[T], T],
        abs_func: Callable[[T], T],
        where_func: Callable[[T, T, T], T],
        convert_func: Callable[..., T],
        map_func: Callable[[Callable[[T], T], T], T],
        resize_func: Callable[[T, Tuple[int, int]], T],
        concat_func: Callable[[List[T], int], T],
        logical_and_func: Callable[[T, T], T],
        mask_func: Callable[[T, T], T],
        split_func: Callable[[T, T, int], List[T]],
        reshape_func: Callable[[T, Tuple[int, int]], T]
) -> Tuple[T, List[T]]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    offset_fractions: array of floats in [0.0, 1.0).
    """
    images_shape = shape_func(images)
    assert len(images_shape) == 4
    assert images_shape[0] == len(offset_fractions)

    image_height, image_width = images_shape[1:3]

    # Positive means the image move downward w.r.t. the original.
    offset_heights = rint_func(multiply_func(offset_fractions, image_height))
    # Positive means the image move to the right w.r.t. the original.
    offset_widths = rint_func(multiply_func(offset_fractions, image_width))

    cropped_image_heights = image_height - abs_func(offset_heights)
    cropped_image_widths = image_width - abs_func(offset_widths)

    tops = where_func(offset_heights > 0, 0, -offset_heights)
    lefts = where_func(offset_widths > 0, 0, -offset_widths)
    bottoms = tops + cropped_image_heights
    rights = lefts + cropped_image_widths

    image_idxes = list(range(images_shape[0]))

    def crop(p: T) -> T:
        """
        p: (image_idx, top, left, bottom, right)
        """
        return images[p[0], p[1]:p[3], p[2]:p[4], :]

    image_param = convert_func(list(zip(
        image_idxes, tops, lefts, bottoms, rights
    )))
    cropped_images = map_func(crop, image_param)

    images = resize_func(cropped_images, (image_height, image_width))
    assert shape_func(images)[1:3] == (image_height, image_width)

    def make_bboxes(p: T) -> T:
        """
        p: (image_idx, top, left, cropped_image_width, cropped_image_height)
        """
        idx, top, left, cropped_image_width, cropped_image_height = p
        bboxes = bboxes_list[idx]

        # Translation.
        xs = bboxes[:, :1] - left
        ys = bboxes[:, 1:2] - top

        # Resizing.
        w = image_width / cropped_image_width
        h = image_height / cropped_image_height
        xs = xs * w
        widths = bboxes[:, 2:3] * w
        ys = ys * h
        heights = bboxes[:, 3:] * h

        bboxes = concat_func([xs, ys, widths, heights], 1)

        # Excluding bboxes out of image.
        xmaxs = xs + widths
        ymaxs = ys + heights
        included = logical_and_func(
            logical_and_func(xs >= 0, xmaxs <= image_width),
            logical_and_func(ys >= 0, ymaxs <= image_height)
        )
        return mask_func(bboxes, included)

    bboxes_param = convert_func(list(zip(
        image_idxes, tops, lefts, cropped_image_widths, cropped_image_heights
    )))
    all_bboxes = map_func(make_bboxes, bboxes_param)

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    bboxes_list = _make_output_bboxes_list(
        bboxes_nums, all_bboxes, split_func, reshape_func
    )

    return images, bboxes_list
