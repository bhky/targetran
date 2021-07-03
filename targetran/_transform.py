"""
Image and target transform utilities.
"""

from typing import Callable, List, Tuple, TypeVar

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

from ._functional import (
    _map_single,
    _np_convert,
    _np_ragged_to_list,
    _np_list_to_ragged,
    _np_stack_bboxes,
    _np_resize_image,
    _np_boolean_mask,
    _np_multiply,
    _np_logical_and,
    _np_pad_images,
    _np_make_bboxes_ragged,
    _tf_convert,
    _tf_ragged_to_list,
    _tf_list_to_ragged,
    _tf_stack_bboxes,
    _tf_resize_image,
    _tf_pad_images,
    _tf_make_bboxes_ragged
)


T = TypeVar("T", np.ndarray, tf.Tensor)


def _flip_left_right(
        images: T,
        bboxes_ragged: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        convert_fn: Callable[..., T],
        stack_bboxes_fn: Callable[[T], T],
        concat_fn: Callable[[List[T], int], T],
        make_bboxes_ragged_fn: Callable[[T, T], T]
) -> Tuple[T, T]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    image_width = convert_fn(images_shape[2])

    images = images[..., ::-1, :]

    all_bboxes = stack_bboxes_fn(bboxes_ragged)

    xs = image_width - all_bboxes[:, :1] - all_bboxes[:, 2:3]
    all_bboxes = concat_fn(
        [xs, all_bboxes[:, 1:]], 1  # Along axis 1.
    )

    bboxes_ragged = make_bboxes_ragged_fn(all_bboxes, bboxes_ragged)

    return images, bboxes_ragged


def _flip_up_down(
        images: T,
        bboxes_ragged: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        convert_fn: Callable[..., T],
        stack_bboxes_fn: Callable[[T], T],
        concat_fn: Callable[[List[T], int], T],
        make_bboxes_ragged_fn: Callable[[T, T], T]
) -> Tuple[T, T]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    image_height = convert_fn(images_shape[1])

    images = images[:, ::-1, ...]

    all_bboxes = stack_bboxes_fn(bboxes_ragged)

    ys = image_height - all_bboxes[:, 1:2] - all_bboxes[:, 3:]
    all_bboxes = concat_fn(
        [all_bboxes[:, :1], ys, all_bboxes[:, 2:]], 1  # Along axis 1.
    )

    bboxes_ragged = make_bboxes_ragged_fn(all_bboxes, bboxes_ragged)

    return images, bboxes_ragged


def _rotate_90(
        images: T,
        bboxes_ragged: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        convert_fn: Callable[..., T],
        transpose_fn: Callable[[T, Tuple[int, ...]], T],
        stack_bboxes_fn: Callable[[T], T],
        concat_fn: Callable[[List[T], int], T],
        make_bboxes_ragged_fn: Callable[[T, T], T]
) -> Tuple[T, T]:
    """
    Rotate 90 degrees anti-clockwise.

    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    image_width = convert_fn(images_shape[2])

    images = transpose_fn(images, (0, 2, 1, 3))[:, ::-1, :, :]

    all_bboxes = stack_bboxes_fn(bboxes_ragged)

    all_bboxes = concat_fn([
        all_bboxes[:, 1:2],
        image_width - all_bboxes[:, :1] - all_bboxes[:, 2:3],
        all_bboxes[:, 3:],
        all_bboxes[:, 2:3],
    ], 1)  # Along axis 1.

    bboxes_ragged = make_bboxes_ragged_fn(all_bboxes, bboxes_ragged)

    return images, bboxes_ragged


def _pad(
        images: T,
        bboxes_ragged: T,
        pad_offsets: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        pad_images_fn: Callable[[T, T], T],
        stack_bboxes_fn: Callable[[T], T],
        concat_fn: Callable[[List[T], int], T],
        make_bboxes_ragged_fn: Callable[[T, T], T]
) -> Tuple[T, T]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    pad_offsets: (top, bottom, left, right)
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    images = pad_images_fn(images, pad_offsets)

    all_bboxes = stack_bboxes_fn(bboxes_ragged)

    all_bboxes = concat_fn([
        all_bboxes[:, :1] + int(pad_offsets[2]),
        all_bboxes[:, 1:2] + int(pad_offsets[0]),
        all_bboxes[:, 2:3],
        all_bboxes[:, 3:],
    ], 1)  # Along axis 1.

    bboxes_ragged = make_bboxes_ragged_fn(all_bboxes, bboxes_ragged)

    return images, bboxes_ragged


def _rotate_90_and_pad(
        images: T,
        bboxes_ragged: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        convert_fn: Callable[..., T],
        transpose_fn: Callable[[T, Tuple[int, ...]], T],
        stack_bboxes_fn: Callable[[T], T],
        concat_fn: Callable[[List[T], int], T],
        where_fn: Callable[[T, T, T], T],
        ceil_fn: Callable[[T], T],
        floor_fn: Callable[[T], T],
        pad_images_fn: Callable[[T, T], T],
        make_bboxes_ragged_fn: Callable[[T, T], T]
) -> Tuple[T, T]:
    """
    Rotate 90 degrees anti-clockwise and *try* to pad to the same aspect ratio.

    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    images, bboxes_ragged = _rotate_90(
        images, bboxes_ragged,
        shape_fn, convert_fn, transpose_fn, stack_bboxes_fn, concat_fn,
        make_bboxes_ragged_fn
    )
    new_height = convert_fn(shape_fn(images)[1])
    new_width = convert_fn(shape_fn(images)[2])
    longer = where_fn(new_height > new_width, new_height, new_width)
    shorter = where_fn(new_height < new_width, new_height, new_width)

    pad_length = longer ** 2 / shorter - shorter
    half_pad_length = pad_length / convert_fn(2.0)
    pad_major = int(ceil_fn(half_pad_length))
    pad_minor = int(floor_fn(half_pad_length))

    pad_offsets = where_fn(
        new_height > new_width,
        convert_fn((0, 0, pad_major, pad_minor)),
        convert_fn((pad_major, pad_minor, 0, 0))
    )
    return _pad(
        images, bboxes_ragged, pad_offsets, shape_fn, pad_images_fn,
        stack_bboxes_fn, concat_fn, make_bboxes_ragged_fn
    )


def _resize_single(
        image: T,
        bboxes: T,
        dest_size: Tuple[int, int],
        shape_fn: Callable[[T], Tuple[int, ...]],
        resize_image_fn: Callable[[T, Tuple[int, int]], T],
        convert_fn: Callable[..., T],
        concat_fn: Callable[[List[T], int], T],
) -> Tuple[T, T]:
    """
    image: [h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    dest_size: (height, width)
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3

    image = resize_image_fn(image, dest_size)

    w = convert_fn(dest_size[1] / image_shape[1])
    h = convert_fn(dest_size[0] / image_shape[0])

    xs = bboxes[:, :1] * w
    ys = bboxes[:, 1:2] * h
    widths = bboxes[:, 2:3] * w
    heights = bboxes[:, 3:] * h
    bboxes = concat_fn([xs, ys, widths, heights], 1)

    return image, bboxes


def _crop_single(
        image: T,
        bboxes: T,
        x_offset_fraction: float,
        y_offset_fraction: float,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, int]], T],
        convert_fn: Callable[..., T],
        multiply_fn: Callable[[T, T], T],
        rint_fn: Callable[[T], T],
        abs_fn: Callable[[T], T],
        where_fn: Callable[[T, T, T], T],
        concat_fn: Callable[[List[T], int], T],
        logical_and_fn: Callable[[T, T], T],
        squeeze_fn: Callable[[T, int], T],
        boolean_mask_fn: Callable[[T, T], T],
) -> Tuple[T, T]:
    """
    image: [h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    offset_fraction: float in [0.0, 1.0)
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3

    image_height, image_width = image_shape[0:2]
    image_height = convert_fn(image_height)
    image_width = convert_fn(image_width)
    x_offset_fraction = convert_fn(x_offset_fraction)
    y_offset_fraction = convert_fn(y_offset_fraction)

    # Positive means the image move downward w.r.t. the original.
    offset_height = rint_fn(multiply_fn(y_offset_fraction, image_height))
    # Positive means the image move to the right w.r.t. the original.
    offset_width = rint_fn(multiply_fn(x_offset_fraction, image_width))

    cropped_image_height = image_height - abs_fn(offset_height)
    cropped_image_width = image_width - abs_fn(offset_width)

    top = where_fn(offset_height > 0, convert_fn(0), -offset_height)
    left = where_fn(offset_width > 0, convert_fn(0), -offset_width)
    bottom = top + cropped_image_height
    right = left + cropped_image_width

    # Crop image.
    image = image[int(top):int(bottom), int(left):int(right), :]

    # Translate bboxes.
    bboxes = reshape_fn(bboxes, (-1, 4))
    xs = bboxes[:, :1] - offset_width
    ys = bboxes[:, 1:2] - offset_height
    widths = bboxes[:, 2:3]
    heights = bboxes[:, 3:]
    bboxes = concat_fn([xs, ys, widths, heights], 1)

    # Filter bboxes.
    xmaxs = xs + widths
    ymaxs = ys + heights
    included = squeeze_fn(logical_and_fn(
        logical_and_fn(xs >= 0, xmaxs <= image_width),
        logical_and_fn(ys >= 0, ymaxs <= image_height)
    ), -1)  # Squeeze along the last axis.
    bboxes = boolean_mask_fn(bboxes, included)

    return image, bboxes


def _np_flip_left_right(
        images: np.ndarray,
        bboxes_ragged: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return _flip_left_right(
        images, bboxes_ragged,
        np.shape, _np_convert, _np_stack_bboxes, np.concatenate,
        _np_make_bboxes_ragged
    )


def _np_flip_up_down(
        images: np.ndarray,
        bboxes_ragged: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return _flip_up_down(
        images, bboxes_ragged,
        np.shape, _np_convert, _np_stack_bboxes, np.concatenate,
        _np_make_bboxes_ragged
    )


def _np_rotate_90(
        images: np.ndarray,
        bboxes_ragged: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return _rotate_90(
        images, bboxes_ragged,
        np.shape, _np_convert, np.transpose, _np_stack_bboxes, np.concatenate,
        _np_make_bboxes_ragged
    )


def _np_resize(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        dest_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    image_list = [image for image in images]
    bboxes_list = _np_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _resize_single, image_list, bboxes_list, None,
        dest_size, np.shape, _np_resize_image, _np_convert, np.concatenate
    )
    images = _np_convert(image_list)
    bboxes_ragged = _np_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


def _np_rotate_90_and_pad(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return _rotate_90_and_pad(
        images, bboxes_ragged,
        np.shape, _np_convert, np.transpose, _np_stack_bboxes, np.concatenate,
        np.where, np.ceil, np.floor, _np_pad_images,
        _np_make_bboxes_ragged
    )


def _np_rotate_90_and_pad_and_resize(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = int(np.shape(images)[1]), int(np.shape(images)[2])
    images, bboxes_ragged = _np_rotate_90_and_pad(images, bboxes_ragged)
    return _np_resize(images, bboxes_ragged, (height, width))


def _np_crop_and_resize(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        x_offset_fractions: np.ndarray,
        y_offset_fractions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    image_list = [image for image in images]
    bboxes_list = _np_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _crop_single, image_list, bboxes_list,
        [x_offset_fractions, y_offset_fractions],
        np.shape, np.reshape, _np_convert,
        _np_multiply, np.rint, np.abs, np.where, np.concatenate,
        _np_logical_and, np.squeeze, _np_boolean_mask
    )
    image_list, bboxes_list = _map_single(
        _resize_single, image_list, bboxes_list, None,
        np.shape(images)[1:3], np.shape, _np_resize_image,
        _np_convert, np.concatenate
    )
    images = _np_convert(image_list)
    bboxes_ragged = _np_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


def _tf_flip_left_right(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _flip_left_right(
        images, bboxes_ragged,
        tf.shape, _tf_convert, _tf_stack_bboxes, tf.concat,
        _tf_make_bboxes_ragged
    )


def _tf_flip_up_down(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _flip_up_down(
        images, bboxes_ragged,
        tf.shape, _tf_convert, _tf_stack_bboxes, tf.concat,
        _tf_make_bboxes_ragged
    )


def _tf_rotate_90(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _rotate_90(
        images, bboxes_ragged,
        tf.shape, _tf_convert, tf.transpose, _tf_stack_bboxes, tf.concat,
        _tf_make_bboxes_ragged
    )


def _tf_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
        dest_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor]:
    image_list = [image for image in images]
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    image_list, bboxes_ragged = _map_single(
        _resize_single, image_list, bboxes_list, None,
        dest_size, tf.shape, _tf_resize_image, _tf_convert, tf.concat
    )
    images = _tf_convert(image_list)
    bboxes_ragged = _tf_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


def _tf_rotate_90_and_pad(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _rotate_90_and_pad(
        images, bboxes_ragged,
        tf.shape, _tf_convert, tf.transpose, _tf_stack_bboxes, tf.concat,
        tf.where, tf.math.ceil, tf.math.floor, _tf_pad_images,
        _tf_make_bboxes_ragged
    )


def _tf_rotate_90_and_pad_and_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    height, width = int(tf.shape(images)[1]), int(tf.shape(images)[2])
    images, bboxes_ragged = _tf_rotate_90_and_pad(images, bboxes_ragged)
    return _tf_resize(images, bboxes_ragged, (height, width))


def _tf_crop_and_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
        x_offset_fractions: tf.Tensor,
        y_offset_fractions: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    image_list = [image for image in images]
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _crop_single, image_list, bboxes_list,
        [x_offset_fractions, y_offset_fractions],
        tf.shape, tf.reshape, _tf_convert,
        tf.multiply, tf.math.rint, tf.abs, tf.where, tf.concat,
        tf.logical_and, tf.squeeze, tf.boolean_mask
    )
    image_list, bboxes_list = _map_single(
        _resize_single, image_list, bboxes_list, None,
        tf.shape(images)[1:3], tf.shape, _tf_resize_image,
        _tf_convert, tf.concat
    )
    images = _tf_convert(image_list)
    bboxes_ragged = _tf_list_to_ragged(bboxes_list)
    return images, bboxes_ragged
