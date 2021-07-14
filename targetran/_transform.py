"""
Image and target transform utilities.
"""

from typing import Callable, List, Tuple, TypeVar

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore


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


def _rotate_single(
        image: T,
        bboxes: T,
        angle_deg: float,
        shape_fn: Callable[[T], Tuple[int, ...]],
        convert_fn: Callable[..., T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        pad_images_fn: Callable[[T, T], T],
        range_fn: Callable[[int, int, int], T],
        round_to_int_fn: Callable[[T], T],
        repeat_fn: Callable[[T, T], T],
        tile_fn: Callable[[T, T], T],
        stack_fn: Callable[[List[T], int], T],
        concat_fn: Callable[[List[T], int], T],
        cos_fn: Callable[[T], T],
        sin_fn: Callable[[T], T],
        matmul_fn: Callable[[T, T], T],
        clip_fn: Callable[[T, T, T], T],
        gather_image_fn: Callable[[T, T], T],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        copy_fn: Callable[[T], T],
        max_fn: Callable[[T, int], T],
        min_fn: Callable[[T, int], T],
        logical_and_fn: Callable[[T, T], T],
        boolean_mask_fn: Callable[[T, T], T]
) -> Tuple[T, T]:
    """
    image: [h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3

    height, width = int(image_shape[0]), int(image_shape[1])
    num_channels = int(image_shape[2])

    # Pad image to provide a zero-value pixel frame for clipping use below.
    pad_offsets = convert_fn([1, 1, 1, 1])
    image = squeeze_fn(pad_images_fn(expand_dim_fn(image, 0), pad_offsets), 0)

    # References:
    # https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96

    # Destination indices. Note that (-foo // 2) != -(foo // 2).
    row_idxes = repeat_fn(  # Along y-axis, from top to bottom.
        round_to_int_fn(range_fn(-(height // 2), height // 2 + 1, 1)),
        round_to_int_fn(convert_fn([height]))
    )
    col_idxes = tile_fn(  # Along x-axis, from left to right.
        round_to_int_fn(range_fn(-(width // 2), width // 2 + 1, 1)),
        round_to_int_fn(convert_fn([width]))
    )
    # Note the (col, row) -> (x, y) swapping.
    image_idxes = stack_fn([col_idxes, row_idxes], 0)

    # Rotation matrix. Clockwise for the indices, so the final image would
    # appear to be rotated anti-clockwise.
    ang_rad = convert_fn(np.pi * angle_deg / 180.0)
    image_rot_mat = convert_fn([
        [cos_fn(ang_rad), -sin_fn(ang_rad)],
        [sin_fn(ang_rad), cos_fn(ang_rad)]
    ])
    new_image_idxes = matmul_fn(image_rot_mat, image_idxes)
    clipped_new_image_idxes = clip_fn(
        new_image_idxes,
        # Note the extra idx for the padded frame.
        convert_fn([
            [-(width // 2) - 1], [-(height // 2) - 1]
        ]),
        convert_fn([
            [width // 2 + 2], [height // 2 + 2]
        ])
    )

    # Assigning original pixel values to new positions.
    orig_image_idxes = concat_fn([
        clipped_new_image_idxes[1:] + height // 2 + 1,  # Rows.
        clipped_new_image_idxes[:1] + width // 2 + 1  # Columns.
    ], 0)
    orig_image_idxes = round_to_int_fn(orig_image_idxes)
    values = gather_image_fn(image, orig_image_idxes)
    new_image = reshape_fn(values, (height, width, num_channels))

    # Transform bboxes.
    top_left_xs = bboxes[:, :1]
    top_left_ys = bboxes[:, 1:2]
    top_right_xs = bboxes[:, :1] + bboxes[:, 2:3]
    top_right_ys = bboxes[:, 1:2]
    bottom_left_xs = copy_fn(top_left_xs)
    bottom_left_ys = copy_fn(top_left_ys + bboxes[:, 3:])
    bottom_right_xs = copy_fn(top_right_xs)
    bottom_right_ys = copy_fn(top_right_ys + bboxes[:, 3:])

    xs = concat_fn(
        [top_left_xs - width // 2,
         top_right_xs - width // 2,
         bottom_left_xs - width // 2,
         bottom_right_xs - width // 2],
        1
    )
    ys = concat_fn(
        [top_left_ys - height // 2,
         top_right_ys - height // 2,
         bottom_left_ys - height // 2,
         bottom_right_ys - height // 2],
        1
    )
    bboxes_idxes = stack_fn([xs, ys], 1)  # Shape: [num_bboxes, 2, 4].

    bboxes_rot_mat = convert_fn([  # Anti-clockwise.
        [cos_fn(ang_rad), sin_fn(ang_rad)],
        [-sin_fn(ang_rad), cos_fn(ang_rad)]
    ])
    new_bboxes_idxes = matmul_fn(bboxes_rot_mat, bboxes_idxes)

    # New bboxes, defined as the rectangle enclosing the transformed bboxes.
    new_xs = new_bboxes_idxes[:, 0, :]  # Shape: [num_bboxes, 4].
    new_ys = new_bboxes_idxes[:, 1, :]
    max_xs = max_fn(new_xs, -1)  # Shape: [num_bboxes].
    max_ys = max_fn(new_ys, -1)
    min_xs = min_fn(new_xs, -1)
    min_ys = min_fn(new_ys, -1)

    new_top_left_xs = expand_dim_fn(min_xs, -1)
    new_top_left_ys = expand_dim_fn(min_ys, -1)
    new_bottom_right_xs = expand_dim_fn(max_xs, -1)
    new_bottom_right_ys = expand_dim_fn(max_ys, -1)
    new_widths = new_bottom_right_xs - new_top_left_xs
    new_heights = new_bottom_right_ys - new_top_left_ys
    new_bboxes = concat_fn([  # Shape: [num_bboxes, 4].
        new_top_left_xs, new_top_left_ys, new_widths, new_heights
    ], -1)

    # Filter new bboxes.
    included = logical_and_fn(
        logical_and_fn(
            min_xs >= -(width // 2), max_xs <= width // 2 + 1
        ),
        logical_and_fn(
            min_ys >= -(height // 2), max_ys <= height // 2 + 1
        )
    )
    new_bboxes = boolean_mask_fn(new_bboxes, included)

    new_bboxes = concat_fn([
        new_bboxes[:, :1] + width // 2,
        new_bboxes[:, 1:2] + height // 2,
        new_bboxes[:, 2:3],
        new_bboxes[:, 3:]
    ], 1)

    return new_image, new_bboxes


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


def _get_random_size_fractions(
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., T],
        convert_fn: Callable[..., T],
) -> Tuple[T, T]:
    """
    height_fraction_range, width_fraction_range: (-1.0, 1.0)
    rand_fn: generate random [0.0, 1.0) array of batch size
    """
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

    return height_fractions, width_fractions


def _get_random_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., T],
        convert_fn: Callable[..., T],
        round_to_int_fn: Callable[[T], T]
) -> Tuple[T, T, T, T]:
    """
    height_fraction_range, width_fraction_range: [0.0, 1.0)
    rand_fn: generate random [0.0, 1.0) array of batch size
    Return: randomized (offset_heights, offset_widths,
                        cropped_image_heights, cropped_image_widths)
    """
    image_height = convert_fn(image_height)
    image_width = convert_fn(image_width)
    height_fraction_range = convert_fn(height_fraction_range)
    width_fraction_range = convert_fn(width_fraction_range)

    height_fractions, width_fractions = _get_random_size_fractions(
        height_fraction_range, width_fraction_range, rand_fn, convert_fn
    )

    cropped_image_heights = image_height * height_fractions
    cropped_image_widths = image_height * width_fractions

    offset_heights = round_to_int_fn(
        (image_height - cropped_image_heights) * rand_fn()
    )
    offset_widths = round_to_int_fn(
        (image_width - cropped_image_widths) * rand_fn()
    )

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
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
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
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        convert_fn: Callable[..., T],
        where_fn: Callable[[T, T, T], T],
        abs_fn: Callable[[T], T],
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
        convert_fn([-translate_height, 0, -translate_height])
    )
    offset_width, pad_left, pad_right = where_fn(
        translate_width >= 0,
        convert_fn([0, translate_width, 0]),
        convert_fn([-translate_width, 0, -translate_width])
    )

    cropped_height = convert_fn(image_shape[0]) - abs_fn(translate_height)
    cropped_width = convert_fn(image_shape[1]) - abs_fn(translate_width)

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
