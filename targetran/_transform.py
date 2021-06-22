"""
Image and target transform utilities.
"""

from typing import Callable, List, Tuple, TypeVar

import numpy as np
import tensorflow as tf

T = TypeVar("T", np.ndarray, tf.Tensor, float)


def _reshape_bboxes(
        bboxes_list: List[T],
        reshape_fn: Callable[[T, Tuple[int, int]], T]
) -> List[T]:
    """
    This seemingly extra process is mainly for tackling empty bboxes array.
    """
    return [reshape_fn(bboxes, (-1, 4)) for bboxes in bboxes_list]


def _make_bboxes_list(
        bboxes_nums: List[int],
        all_bboxes: T,
        split_fn: Callable[[T, T, int], List[T]],
        reshape_fn: Callable[[T, Tuple[int, int]], T]
) -> List[T]:
    """
    Helper function for splitting all_bboxes array to list of bboxes.
    """
    indices = np.cumsum(bboxes_nums)[:-1]
    bboxes_list = split_fn(all_bboxes, indices, 0)  # Along axis 0.
    return _reshape_bboxes(bboxes_list, reshape_fn)


def _flip_left_right(
        images: T,
        bboxes_list: List[T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        concat_fn: Callable[[List[T], int], T],
        split_fn: Callable[[T, T, int], List[T]],
        reshape_fn: Callable[[T, Tuple[int, int]], T]
) -> Tuple[T, List[T]]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    image_width = images_shape[2]

    images = images[..., ::-1, :]

    bboxes_list = _reshape_bboxes(bboxes_list, reshape_fn)
    all_bboxes = concat_fn(bboxes_list, 0)  # Along axis 0.
    assert shape_fn(all_bboxes)[-1] == 4

    all_bboxes[:, :1] = image_width - all_bboxes[:, :1] - all_bboxes[:, 2:3]

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    bboxes_list = _make_bboxes_list(
        bboxes_nums, all_bboxes, split_fn, reshape_fn
    )

    return images, bboxes_list


def _flip_up_down(
        images: T,
        bboxes_list: List[T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        concat_fn: Callable[[List[T], int], T],
        split_fn: Callable[[T, T, int], List[T]],
        reshape_fn: Callable[[T, Tuple[int, int]], T]
) -> Tuple[T, List[T]]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    image_height = images_shape[1]

    images = images[:, ::-1, ...]

    bboxes_list = _reshape_bboxes(bboxes_list, reshape_fn)
    all_bboxes = concat_fn(bboxes_list, 0)  # Along axis 0.
    assert shape_fn(all_bboxes)[-1] == 4

    all_bboxes[:, 1:2] = image_height - all_bboxes[:, 1:2] - all_bboxes[:, 3:]

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    bboxes_list = _make_bboxes_list(
        bboxes_nums, all_bboxes, split_fn, reshape_fn
    )

    return images, bboxes_list


def _rotate_90(
        images: np.ndarray,
        bboxes_list: List[np.ndarray],
        shape_fn: Callable[[T], Tuple[int, ...]],
        transpose_fn: Callable[[T, Tuple[int, ...]], T],
        concat_fn: Callable[[List[T], int], T],
        split_fn: Callable[[T, T, int], List[T]],
        reshape_fn: Callable[[T, Tuple[int, int]], T]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Rotate 90 degrees anti-clockwise.
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    image_width = images_shape[2]

    images = transpose_fn(images, (0, 2, 1, 3))[:, ::-1, :, :]

    bboxes_list = _reshape_bboxes(bboxes_list, reshape_fn)
    all_bboxes = concat_fn(bboxes_list, 0)  # Along axis 0.
    assert shape_fn(all_bboxes)[-1] == 4

    all_bboxes = concat_fn([
        all_bboxes[:, 1:2],
        image_width - all_bboxes[:, :1] - all_bboxes[:, 2:3],
        all_bboxes[:, 3:],
        all_bboxes[:, 2:3],
    ], 1)  # Along axis 1.

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    bboxes_list = _make_bboxes_list(
        bboxes_nums, all_bboxes, split_fn, reshape_fn
    )

    return images, bboxes_list


def _crop_and_resize(
        images: T,
        bboxes_list: List[T],
        x_offset_fractions: T,
        y_offset_fractions: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        multiply_fn: Callable[[T, T], T],
        rint_fn: Callable[[T], T],
        abs_fn: Callable[[T], T],
        where_fn: Callable[[T, T, T], T],
        convert_fn: Callable[..., T],
        map_fn: Callable[[Callable[[T], T], T], T],
        resize_fn: Callable[[T, Tuple[int, int]], T],
        concat_fn: Callable[[List[T], int], T],
        logical_and_fn: Callable[[T, T], T],
        squeeze_fn: Callable[[T], T],
        boolean_mask_fn: Callable[[T, T], T],
        split_fn: Callable[[T, T, int], List[T]],
        reshape_fn: Callable[[T, Tuple[int, int]], T]
) -> Tuple[T, List[T]]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    offset_fractions: array of floats in [0.0, 1.0).
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4
    assert images_shape[0] == len(x_offset_fractions) == len(y_offset_fractions)

    image_height, image_width = images_shape[1:3]

    # Positive means the image move downward w.r.t. the original.
    offset_heights = rint_fn(multiply_fn(y_offset_fractions, image_height))
    # Positive means the image move to the right w.r.t. the original.
    offset_widths = rint_fn(multiply_fn(x_offset_fractions, image_width))

    cropped_image_heights = image_height - abs_fn(offset_heights)
    cropped_image_widths = image_width - abs_fn(offset_widths)

    tops = where_fn(offset_heights > 0, 0, -offset_heights)
    lefts = where_fn(offset_widths > 0, 0, -offset_widths)
    bottoms = tops + cropped_image_heights
    rights = lefts + cropped_image_widths

    image_idxes = list(range(images_shape[0]))

    def crop(p: T) -> T:
        """
        p: (image_idx, top, left, bottom, right)
        """
        return images[
           int(p[0]),
           int(p[1]):int(p[3]),
           int(p[2]):int(p[4]),
           :
        ]

    image_param = convert_fn(list(zip(
        image_idxes, tops, lefts, bottoms, rights
    )))
    cropped_images = map_fn(crop, image_param)

    images = resize_fn(cropped_images, (image_height, image_width))
    assert shape_fn(images)[1:3] == (image_height, image_width)

    bboxes_list = _reshape_bboxes(bboxes_list, reshape_fn)

    def make_bboxes(p: T) -> T:
        """
        p: (image_idx, offset_height, offset_width,
            cropped_image_width, cropped_image_height)
        """
        idx, h_offset, w_offset, cropped_image_width, cropped_image_height = p
        bboxes = bboxes_list[int(idx)]

        # Translation.
        xs = bboxes[:, :1] - w_offset
        ys = bboxes[:, 1:2] - h_offset

        # Resizing.
        w = image_width / cropped_image_width
        h = image_height / cropped_image_height
        xs = xs * w
        widths = bboxes[:, 2:3] * w
        ys = ys * h
        heights = bboxes[:, 3:] * h

        return concat_fn([xs, ys, widths, heights], 1)

    def filter_bboxes(bboxes: T) -> T:
        """
        Excluding bboxes out of image.
        """
        xs = bboxes[:, :1]
        ys = bboxes[:, 1:2]
        widths = bboxes[:, 2:3]
        heights = bboxes[:, 3:]
        xmaxs = xs + widths
        ymaxs = ys + heights
        included = squeeze_fn(logical_and_fn(
            logical_and_fn(xs >= 0, xmaxs <= image_width),
            logical_and_fn(ys >= 0, ymaxs <= image_height)
        ))
        return boolean_mask_fn(bboxes, included)

    bboxes_param = convert_fn(list(zip(
        image_idxes, offset_heights, offset_widths,
        cropped_image_widths, cropped_image_heights
    )))
    all_bboxes = map_fn(make_bboxes, bboxes_param)



    # todo: fix
    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    bboxes_list = _make_bboxes_list(
        bboxes_nums, all_bboxes, split_fn, reshape_fn
    )

    return images, bboxes_list
