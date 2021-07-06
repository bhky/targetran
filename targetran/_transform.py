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


def _translate_bboxes(
        bboxes: T,
        top_offset: int,
        left_offset: int,
        concat_fn: Callable[[List[T], int], T]
) -> T:
    return concat_fn([
        bboxes[:, :1] + left_offset,
        bboxes[:, 1:2] + top_offset,
        bboxes[:, 2:3],
        bboxes[:, 3:],
    ], 1)  # Along axis 1.


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

    all_bboxes = _translate_bboxes(
        all_bboxes, int(pad_offsets[0]), int(pad_offsets[2]), concat_fn
    )

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


def _get_random_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., T],
        convert_fn: Callable[..., T],
        rint_fn: Callable[[T], T]
) -> Tuple[T, T, T, T]:
    """
    rand_fn: generate random [0.0, 1.0) array of batch size.
    Return: randomized (offset_heights, offset_widths,
                        cropped_image_heights, cropped_image_widths)
    """
    image_height = convert_fn(image_height)
    image_width = convert_fn(image_width)
    height_fraction_range = convert_fn(height_fraction_range)
    width_fraction_range = convert_fn(width_fraction_range)

    min_height_fraction = height_fraction_range[0]
    min_width_fraction = width_fraction_range[0]
    height_fraction_diff = height_fraction_range[1] - min_height_fraction
    width_fraction_diff = width_fraction_range[1] - min_width_fraction
    assert height_fraction_diff > 0
    assert width_fraction_diff > 0

    height_fractions = height_fraction_diff * rand_fn() + min_height_fraction
    width_fractions = width_fraction_diff * rand_fn() + min_width_fraction

    cropped_image_heights = image_height * height_fractions
    cropped_image_widths = image_height * width_fractions

    offset_heights = rint_fn((image_height - cropped_image_heights) * rand_fn())
    offset_widths = rint_fn((image_width - cropped_image_widths) * rand_fn())

    return (
        offset_heights, offset_widths,
        cropped_image_heights, cropped_image_widths
    )


def _crop_single(
        image: T,
        bboxes: T,
        offset_height: int,
        offset_width: int,
        cropped_image_height: int,
        cropped_image_width: int,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, int]], T],
        convert_fn: Callable[..., T],
        concat_fn: Callable[[List[T], int], T],
        logical_and_fn: Callable[[T, T], T],
        squeeze_fn: Callable[[T, int], T],
        boolean_mask_fn: Callable[[T, T], T],
) -> Tuple[T, T]:
    """
    image: [h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    offset_height: [0, image_height - 1]
    offset_width: [0, image_width - 1]
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3

    offset_height = convert_fn(offset_height)
    offset_width = convert_fn(offset_width)
    cropped_image_height = convert_fn(cropped_image_height)
    cropped_image_width = convert_fn(cropped_image_width)

    top = offset_height
    left = offset_width
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
        logical_and_fn(xs >= 0, xmaxs <= cropped_image_width),
        logical_and_fn(ys >= 0, ymaxs <= cropped_image_height)
    ), -1)  # Squeeze along the last axis.
    bboxes = boolean_mask_fn(bboxes, included)

    return image, bboxes


def _translate_single(
        image: T,
        bboxes: T,
        translate_height: int,
        translate_width: int,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, int]], T],
        convert_fn: Callable[..., T],
        where_fn: Callable[[T, T, T], T],
        concat_fn: Callable[[List[T], int], T],
        logical_and_fn: Callable[[T, T], T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        boolean_mask_fn: Callable[[T, T], T],
        pad_images_fn: Callable[[T, T], T],
) -> Tuple[T, T]:
    """
    Making use of cropping and padding to perform translation.

    image: [h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    translate_height: [-image_height + 1: image_height - 1]
    translate_width: [-image_width + 1: image_width - 1]
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3

    translate_height = convert_fn(translate_height)
    translate_width = convert_fn(translate_width)

    offset_height, pad_top, pad_bottom = where_fn(
        translate_height >= 0,
        convert_fn([0, translate_height, 0]),
        convert_fn([translate_height, 0, translate_height])
    )
    offset_width, pad_left, pad_right = where_fn(
        translate_width >= 0,
        convert_fn([0, translate_width, 0]),
        convert_fn([translate_width, 0, translate_width])
    )

    cropped_height = image_shape[1] - translate_height
    cropped_width = image_shape[2] - translate_width

    image, bboxes = _crop_single(
        image, bboxes,
        offset_height, offset_width, cropped_height, cropped_width,
        shape_fn, reshape_fn, convert_fn, concat_fn, logical_and_fn,
        squeeze_fn, boolean_mask_fn
    )

    image = squeeze_fn(pad_images_fn(
        expand_dim_fn(image, 0),
        convert_fn([pad_top, pad_bottom, pad_left, pad_right])
    ), 0)

    bboxes = _translate_bboxes(bboxes, int(pad_top), int(pad_left), concat_fn)

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


def _np_get_random_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _get_random_crop_inputs(
        image_height, image_width, height_fraction_range, width_fraction_range,
        rand_fn, _np_convert, np.rint
    )


def _np_crop_and_resize(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        offset_heights: np.ndarray,
        offset_widths: np.ndarray,
        cropped_image_heights: np.ndarray,
        cropped_image_widths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    image_list = [image for image in images]
    bboxes_list = _np_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _crop_single, image_list, bboxes_list,
        [offset_heights, offset_widths,
         cropped_image_heights, cropped_image_widths],
        np.shape, np.reshape, _np_convert, np.concatenate,
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


def _np_translate(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        translate_heights: np.ndarray,
        translate_widths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    image_list = [image for image in images]
    bboxes_list = _np_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _translate_single, image_list, bboxes_list,
        [translate_heights, translate_widths],
        np.shape, np.reshape, _np_convert, np.where, np.concatenate,
        _np_logical_and, np.expand_dims, np.squeeze, _np_boolean_mask,
        _np_pad_images
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


def _tf_get_random_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return _get_random_crop_inputs(
        image_height, image_width, height_fraction_range, width_fraction_range,
        rand_fn, _tf_convert, tf.math.rint
    )


def _tf_crop_and_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
        offset_heights: tf.Tensor,
        offset_widths: tf.Tensor,
        cropped_image_heights: tf.Tensor,
        cropped_image_widths: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    image_list = [image for image in images]
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _crop_single, image_list, bboxes_list,
        [offset_heights, offset_widths,
         cropped_image_heights, cropped_image_widths],
        tf.shape, tf.reshape, _tf_convert, tf.concat,
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


def _tf_translate(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
        translate_heights: tf.Tensor,
        translate_widths: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    image_list = [image for image in images]
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _translate_single, image_list, bboxes_list,
        [translate_heights, translate_widths],
        tf.shape, tf.reshape, _tf_convert, tf.where, tf.concat,
        tf.logical_and, tf.expand_dims, tf.squeeze, tf.boolean_mask,
        _tf_pad_images
    )
    images = _tf_convert(image_list)
    bboxes_ragged = _tf_list_to_ragged(bboxes_list)
    return images, bboxes_ragged
