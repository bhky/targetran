"""
Image and target transform utilities.
"""

from typing import Callable, List, Tuple, TypeVar

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore


T = TypeVar("T", np.ndarray, tf.Tensor)


def _flip_left_right(
        image: T,
        bboxes: T,
        labels: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        convert_fn: Callable[..., T],
        concat_fn: Callable[[List[T], int], T]
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3
    bboxes = reshape_fn(bboxes, (-1, 4))

    image_width = convert_fn(image_shape[1])

    image = image[:, ::-1, :]

    xs = image_width - bboxes[:, :1] - bboxes[:, 2:3]
    bboxes = concat_fn([xs, bboxes[:, 1:]], 1)

    return image, bboxes, labels


def _flip_up_down(
        image: T,
        bboxes: T,
        labels: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        convert_fn: Callable[..., T],
        concat_fn: Callable[[List[T], int], T]
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3
    bboxes = reshape_fn(bboxes, (-1, 4))

    image_height = convert_fn(image_shape[0])

    image = image[::-1, :, :]

    ys = image_height - bboxes[:, 1:2] - bboxes[:, 3:]
    bboxes = concat_fn([bboxes[:, :1], ys, bboxes[:, 2:]], 1)

    return image, bboxes, labels


def _rotate_90(
        image: T,
        bboxes: T,
        labels: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        convert_fn: Callable[..., T],
        transpose_fn: Callable[[T, Tuple[int, ...]], T],
        concat_fn: Callable[[List[T], int], T]
) -> Tuple[T, T, T]:
    """
    Rotate 90 degrees anti-clockwise.

    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3
    bboxes = reshape_fn(bboxes, (-1, 4))

    image_width = convert_fn(image_shape[1])

    image = transpose_fn(image, (1, 0, 2))[::-1, :, :]

    bboxes = concat_fn([
        bboxes[:, 1:2],
        image_width - bboxes[:, :1] - bboxes[:, 2:3],
        bboxes[:, 3:],
        bboxes[:, 2:3],
    ], 1)

    return image, bboxes, labels


def _translate_bboxes(
        bboxes: T,
        top_offset: T,
        left_offset: T,
        concat_fn: Callable[[List[T], int], T]
) -> T:
    return concat_fn([
        bboxes[:, :1] + left_offset,
        bboxes[:, 1:2] + top_offset,
        bboxes[:, 2:3],
        bboxes[:, 3:],
    ], 1)


def _pad(
        image: T,
        bboxes: T,
        labels: T,
        pad_offsets: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        pad_image_fn: Callable[[T, T], T],
        concat_fn: Callable[[List[T], int], T]
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    pad_offsets: (top, bottom, left, right)
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3
    bboxes = reshape_fn(bboxes, (-1, 4))

    image = pad_image_fn(image, pad_offsets)

    bboxes = _translate_bboxes(
        bboxes, pad_offsets[0], pad_offsets[2], concat_fn
    )

    return image, bboxes, labels


def _rotate_90_and_pad(
        image: T,
        bboxes: T,
        labels: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        convert_fn: Callable[..., T],
        transpose_fn: Callable[[T, Tuple[int, ...]], T],
        concat_fn: Callable[[List[T], int], T],
        where_fn: Callable[[T, T, T], T],
        ceil_fn: Callable[[T], T],
        floor_fn: Callable[[T], T],
        pad_image_fn: Callable[[T, T], T]
) -> Tuple[T, T, T]:
    """
    Rotate 90 degrees anti-clockwise and *try* to pad to the same aspect ratio.

    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    """
    image, bboxes, labels = _rotate_90(
        image, bboxes, labels,
        shape_fn, reshape_fn, convert_fn, transpose_fn, concat_fn
    )
    new_height = convert_fn(shape_fn(image)[0])
    new_width = convert_fn(shape_fn(image)[1])
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
        image, bboxes, labels,
        pad_offsets, shape_fn, reshape_fn, pad_image_fn, concat_fn
    )


def _affine_transform(
        image: T,
        bboxes: T,
        labels: T,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        convert_fn: Callable[..., T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        pad_image_fn: Callable[[T, T], T],
        range_fn: Callable[[int, int, int], T],
        round_to_int_fn: Callable[[T], T],
        repeat_fn: Callable[[T, T], T],
        tile_fn: Callable[[T, T], T],
        stack_fn: Callable[[List[T], int], T],
        concat_fn: Callable[[List[T], int], T],
        image_dest_tran_mat: T,
        bboxes_tran_mat: T,
        matmul_fn: Callable[[T, T], T],
        clip_fn: Callable[[T, T, T], T],
        gather_image_fn: Callable[[T, T], T],
        copy_fn: Callable[[T], T],
        max_fn: Callable[[T, int], T],
        min_fn: Callable[[T, int], T],
        logical_and_fn: Callable[[T, T], T],
        boolean_mask_fn: Callable[[T, T], T]
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3
    bboxes = reshape_fn(bboxes, (-1, 4))

    height, width = int(image_shape[0]), int(image_shape[1])
    h_mod, w_mod = height % 2, width % 2
    num_channels = int(image_shape[2])

    # Pad image to provide a zero-value pixel frame for clipping use below.
    pad_offsets = convert_fn([1, 1, 1, 1])
    image = pad_image_fn(image, pad_offsets)

    # References:
    # https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96

    # Destination indices. Note that (-foo // 2) != -(foo // 2).
    row_idxes = repeat_fn(  # Along y-axis, from top to bottom.
        range_fn(-(height // 2) + 1 - h_mod, height // 2 + 1, 1),
        round_to_int_fn(convert_fn([width]))
    )
    col_idxes = tile_fn(  # Along x-axis, from left to right.
        range_fn(-(width // 2) + 1 - w_mod, width // 2 + 1, 1),
        round_to_int_fn(convert_fn([height]))
    )
    # Note the (col, row) -> (x, y) swapping.
    image_dest_idxes = stack_fn([col_idxes, row_idxes], 0)

    # Transform image, with clipping.
    new_image_dest_idxes = matmul_fn(
        image_dest_tran_mat, convert_fn(image_dest_idxes)
    )
    clipped_new_image_dest_idxes = clip_fn(
        new_image_dest_idxes,
        # Note the extra idx for the padded frame.
        convert_fn([
            [-(width // 2) - w_mod], [-(height // 2) - h_mod]
        ]),
        convert_fn([
            [width // 2 + 1], [height // 2 + 1]
        ])
    )

    # Assigning original pixel values to new positions.
    image_orig_idxes = concat_fn([
        # Rows.
        clipped_new_image_dest_idxes[1:] + convert_fn(height // 2 + h_mod),
        # Columns.
        clipped_new_image_dest_idxes[:1] + convert_fn(width // 2 + w_mod)
    ], 0)
    image_orig_idxes = round_to_int_fn(image_orig_idxes)
    values = gather_image_fn(image, image_orig_idxes)
    new_image = reshape_fn(values, (height, width, num_channels))

    # Transform bboxes.
    top_left_xs = bboxes[:, :1]
    top_left_ys = bboxes[:, 1:2]
    top_right_xs = bboxes[:, :1] + bboxes[:, 2:3] - 1
    top_right_ys = bboxes[:, 1:2]
    bottom_left_xs = copy_fn(top_left_xs)
    bottom_left_ys = copy_fn(top_left_ys + bboxes[:, 3:] - 1)
    bottom_right_xs = copy_fn(top_right_xs)
    bottom_right_ys = copy_fn(top_right_ys + bboxes[:, 3:] - 1)

    xs = concat_fn(
        [top_left_xs - convert_fn(width // 2 - 1 + w_mod),
         top_right_xs - convert_fn(width // 2 - 1 + w_mod),
         bottom_left_xs - convert_fn(width // 2 - 1 + w_mod),
         bottom_right_xs - convert_fn(width // 2 - 1 + w_mod)],
        1
    )
    ys = concat_fn(
        [top_left_ys - convert_fn(height // 2 - 1 + h_mod),
         top_right_ys - convert_fn(height // 2 - 1 + h_mod),
         bottom_left_ys - convert_fn(height // 2 - 1 + h_mod),
         bottom_right_ys - convert_fn(height // 2 - 1 + h_mod)],
        1
    )
    bboxes_idxes = stack_fn([xs, ys], 1)  # Shape: [num_bboxes, 2, 4].

    tran_bboxes_idxes = matmul_fn(bboxes_tran_mat, bboxes_idxes)

    # New bboxes, defined as the rectangle enclosing the transformed bboxes.
    tran_xs = tran_bboxes_idxes[:, 0, :]  # Shape: [num_bboxes, 4].
    tran_ys = tran_bboxes_idxes[:, 1, :]
    max_xs = max_fn(tran_xs, -1)  # Shape: [num_bboxes].
    max_ys = max_fn(tran_ys, -1)
    min_xs = min_fn(tran_xs, -1)
    min_ys = min_fn(tran_ys, -1)

    tran_top_left_xs = round_to_int_fn(expand_dim_fn(min_xs, -1))
    tran_top_left_ys = round_to_int_fn(expand_dim_fn(min_ys, -1))
    tran_bottom_right_xs = round_to_int_fn(expand_dim_fn(max_xs, -1))
    tran_bottom_right_ys = round_to_int_fn(expand_dim_fn(max_ys, -1))
    new_widths = tran_bottom_right_xs - tran_top_left_xs + 1
    new_heights = tran_bottom_right_ys - tran_top_left_ys + 1
    tran_bboxes = concat_fn([  # Shape: [num_bboxes, 4].
        tran_top_left_xs, tran_top_left_ys, new_widths, new_heights
    ], -1)

    new_xs = tran_bboxes[:, :1] + width // 2 + w_mod - 1
    new_ys = tran_bboxes[:, 1:2] + height // 2 + h_mod - 1

    # Filter new bboxes values.
    xcens = new_xs + new_widths // 2
    ycens = new_ys + new_heights // 2
    included = squeeze_fn(logical_and_fn(
        logical_and_fn(xcens >= 0, xcens <= width),
        logical_and_fn(ycens >= 0, ycens <= height)
    ), -1)

    # Clip new bboxes values.
    xmaxs = clip_fn(
        convert_fn(new_xs + new_widths), convert_fn(0), convert_fn(width)
    )
    ymaxs = clip_fn(
        convert_fn(new_ys + new_heights), convert_fn(0), convert_fn(height)
    )
    new_xs = clip_fn(convert_fn(new_xs), convert_fn(0), convert_fn(width))
    new_ys = clip_fn(convert_fn(new_ys), convert_fn(0), convert_fn(height))
    new_widths = xmaxs - new_xs
    new_heights = ymaxs - new_ys

    new_bboxes = concat_fn([new_xs, new_ys, new_widths, new_heights], 1)
    new_bboxes = convert_fn(boolean_mask_fn(new_bboxes, included))

    # Filter labels.
    new_labels = boolean_mask_fn(labels, included)

    return new_image, new_bboxes, new_labels


def _rotate(
        image: T,
        bboxes: T,
        labels: T,
        angle_deg: float,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        convert_fn: Callable[..., T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        pad_image_fn: Callable[[T, T], T],
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
        copy_fn: Callable[[T], T],
        max_fn: Callable[[T, int], T],
        min_fn: Callable[[T, int], T],
        logical_and_fn: Callable[[T, T], T],
        boolean_mask_fn: Callable[[T, T], T]
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    angle_deg: goes anti-clockwise.
    """
    if angle_deg == 0.0:
        return image, bboxes, labels

    ang_rad = convert_fn(np.pi * angle_deg / 180.0)

    # Image rotation matrix. Clockwise for the destination indices,
    # so the final image would appear to be rotated anti-clockwise.
    image_dest_rot_mat = convert_fn([
        [cos_fn(ang_rad), -sin_fn(ang_rad)],
        [sin_fn(ang_rad), cos_fn(ang_rad)]
    ])

    bboxes_rot_mat = convert_fn([  # Anti-clockwise.
        [cos_fn(ang_rad), sin_fn(ang_rad)],
        [-sin_fn(ang_rad), cos_fn(ang_rad)]
    ])

    return _affine_transform(
        image, bboxes, labels, shape_fn, reshape_fn, convert_fn,
        expand_dim_fn, squeeze_fn, pad_image_fn, range_fn, round_to_int_fn,
        repeat_fn, tile_fn, stack_fn, concat_fn,
        image_dest_rot_mat, bboxes_rot_mat, matmul_fn,
        clip_fn, gather_image_fn, copy_fn, max_fn, min_fn,
        logical_and_fn, boolean_mask_fn
    )


def _shear(
        image: T,
        bboxes: T,
        labels: T,
        angle_deg: float,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        convert_fn: Callable[..., T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        pad_image_fn: Callable[[T, T], T],
        range_fn: Callable[[int, int, int], T],
        round_to_int_fn: Callable[[T], T],
        repeat_fn: Callable[[T, T], T],
        tile_fn: Callable[[T, T], T],
        stack_fn: Callable[[List[T], int], T],
        concat_fn: Callable[[List[T], int], T],
        tan_fn: Callable[[T], T],
        matmul_fn: Callable[[T, T], T],
        clip_fn: Callable[[T, T, T], T],
        gather_image_fn: Callable[[T, T], T],
        copy_fn: Callable[[T], T],
        max_fn: Callable[[T, int], T],
        min_fn: Callable[[T, int], T],
        logical_and_fn: Callable[[T, T], T],
        boolean_mask_fn: Callable[[T, T], T]
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    angle_deg: goes anti-clockwise, where abs(angle_deg) must be < 90.
    """
    if angle_deg == 0.0:
        return image, bboxes, labels

    ang_rad = convert_fn(np.pi * angle_deg / 180.0)
    factor = tan_fn(ang_rad)

    # Image shear matrix. Clockwise for the destination indices,
    # so the final image would appear to be sheared anti-clockwise.
    image_dest_shear_mat = convert_fn([
        [convert_fn(1), -factor],
        [convert_fn(0), convert_fn(1)]
    ])

    bboxes_shear_mat = convert_fn([  # Anti-clockwise.
        [convert_fn(1), factor],
        [convert_fn(0), convert_fn(1)]
    ])

    return _affine_transform(
        image, bboxes, labels, shape_fn, reshape_fn, convert_fn,
        expand_dim_fn, squeeze_fn, pad_image_fn, range_fn, round_to_int_fn,
        repeat_fn, tile_fn, stack_fn, concat_fn,
        image_dest_shear_mat, bboxes_shear_mat, matmul_fn,
        clip_fn, gather_image_fn, copy_fn, max_fn, min_fn,
        logical_and_fn, boolean_mask_fn
    )


def _resize(
        image: T,
        bboxes: T,
        labels: T,
        dest_size: Tuple[int, int],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        resize_image_fn: Callable[[T, Tuple[int, int]], T],
        convert_fn: Callable[..., T],
        concat_fn: Callable[[List[T], int], T],
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    dest_size: (height, width)
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3
    bboxes = reshape_fn(bboxes, (-1, 4))

    if image_shape[0] == dest_size[0] and image_shape[1] == dest_size[1]:
        return image, bboxes, labels

    image = resize_image_fn(image, dest_size)

    w = convert_fn(dest_size[1] / image_shape[1])
    h = convert_fn(dest_size[0] / image_shape[0])

    xs = bboxes[:, :1] * w
    ys = bboxes[:, 1:2] * h
    widths = bboxes[:, 2:3] * w
    heights = bboxes[:, 3:] * h
    bboxes = concat_fn([xs, ys, widths, heights], 1)

    return image, bboxes, labels


def _get_random_size_fractions(
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., T],
        convert_fn: Callable[..., T],
) -> Tuple[T, T]:
    """
    height_fraction_range, width_fraction_range: (-1.0, 1.0)
    rand_fn: generate array of random numbers in range [0.0, 1.0)
    """
    height_fraction_range = convert_fn(height_fraction_range)
    width_fraction_range = convert_fn(width_fraction_range)

    min_height_fraction = height_fraction_range[0]
    min_width_fraction = width_fraction_range[0]
    height_fraction_diff = height_fraction_range[1] - min_height_fraction
    width_fraction_diff = width_fraction_range[1] - min_width_fraction

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
    height_fraction_range, width_fraction_range: in range [0.0, 1.0)
    rand_fn: generate array of random numbers in range [0.0, 1.0)
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


def _crop(
        image: T,
        bboxes: T,
        labels: T,
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
        clip_fn: Callable[[T, T, T], T],
        boolean_mask_fn: Callable[[T, T], T],
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    offset_height: in range [0, image_height - 1]
    offset_width: in range [0, image_width - 1]
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3
    bboxes = reshape_fn(bboxes, (-1, 4))

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

    # Translate bboxes values.
    xs = bboxes[:, :1] - offset_width
    ys = bboxes[:, 1:2] - offset_height
    widths = bboxes[:, 2:3]
    heights = bboxes[:, 3:]

    # Filter bboxes values.
    xcens = xs + widths // 2
    ycens = ys + heights // 2
    included = squeeze_fn(logical_and_fn(
        logical_and_fn(xcens >= 0, xcens <= cropped_image_width),
        logical_and_fn(ycens >= 0, ycens <= cropped_image_height)
    ), -1)  # Squeeze along the last axis.

    # Clip bboxes values.
    xmaxs = clip_fn(xs + widths, convert_fn(0), cropped_image_width)
    ymaxs = clip_fn(ys + heights, convert_fn(0), cropped_image_height)
    xs = clip_fn(xs, convert_fn(0), cropped_image_width)
    ys = clip_fn(ys, convert_fn(0), cropped_image_height)
    widths = xmaxs - xs
    heights = ymaxs - ys

    bboxes = concat_fn([xs, ys, widths, heights], 1)
    bboxes = boolean_mask_fn(bboxes, included)

    # Filter labels.
    labels = boolean_mask_fn(labels, included)

    return image, bboxes, labels


def _translate(
        image: T,
        bboxes: T,
        labels: T,
        translate_height: int,
        translate_width: int,
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        convert_fn: Callable[..., T],
        where_fn: Callable[[T, T, T], T],
        abs_fn: Callable[[T], T],
        concat_fn: Callable[[List[T], int], T],
        logical_and_fn: Callable[[T, T], T],
        squeeze_fn: Callable[[T, int], T],
        clip_fn: Callable[[T, T, T], T],
        boolean_mask_fn: Callable[[T, T], T],
        pad_image_fn: Callable[[T, T], T],
) -> Tuple[T, T, T]:
    """
    Making use of cropping and padding to perform translation.

    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    translate_height: in range [-image_height + 1: image_height - 1]
    translate_width: in range [-image_width + 1: image_width - 1]
    """
    image_shape = shape_fn(image)
    assert len(image_shape) == 3
    bboxes = reshape_fn(bboxes, (-1, 4))

    translate_height = convert_fn(translate_height)
    translate_width = convert_fn(translate_width)

    t = where_fn(
        translate_height >= 0,
        convert_fn([0, translate_height, 0]),
        convert_fn([-translate_height, 0, -translate_height])
    )
    offset_height, pad_top, pad_bottom = t[0], t[1], t[2]

    t = where_fn(
        translate_width >= 0,
        convert_fn([0, translate_width, 0]),
        convert_fn([-translate_width, 0, -translate_width])
    )
    offset_width, pad_left, pad_right = t[0], t[1], t[2]

    cropped_height = convert_fn(image_shape[0]) - abs_fn(translate_height)
    cropped_width = convert_fn(image_shape[1]) - abs_fn(translate_width)

    image, bboxes, labels = _crop(
        image, bboxes, labels,
        offset_height, offset_width, cropped_height, cropped_width,
        shape_fn, reshape_fn, convert_fn, concat_fn, logical_and_fn,
        squeeze_fn, clip_fn, boolean_mask_fn
    )

    image = pad_image_fn(
        image, convert_fn([pad_top, pad_bottom, pad_left, pad_right])
    )

    bboxes = _translate_bboxes(bboxes, pad_top, pad_left, concat_fn)

    return image, bboxes, labels
