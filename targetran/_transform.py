"""
Image and target transform utilities.
"""
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

from targetran._typing import T
from targetran.utils import Interpolation


def _sanitise(
        image: T,
        bboxes: T,
        labels: T,
        convert_fn: Callable[..., T],
        shape_fn: Callable[[T], Sequence[int]],
        reshape_fn: Callable[[T, Sequence[int]], T],
) -> Tuple[T, T, T]:
    """
    Try to convert input to the expected format.
    """
    if len(shape_fn(image)) != 3:
        raise ValueError(
            "Input image must have 3 dimensions, i.e., "
            "(height, width, num_channels)."
        )
    image = convert_fn(image)
    bboxes = reshape_fn(convert_fn(bboxes), (-1, 4))
    labels = convert_fn(labels)
    return image, bboxes, labels


@dataclass
class _AffineDependency:
    convert_fn: Callable[..., T]
    shape_fn: Callable[[T], Sequence[int]]
    reshape_fn: Callable[[T, Sequence[int]], T]
    expand_dim_fn: Callable[[T, int], T]
    squeeze_fn: Callable[[T, int], T]
    pad_image_fn: Callable[[T, T], T]
    range_fn: Callable[[int, int, int], T]
    cast_to_int_fn: Callable[[T], T]
    round_to_int_fn: Callable[[T], T]
    repeat_fn: Callable[[T, T], T]
    tile_fn: Callable[[T, T], T]
    ones_like_fn: Callable[[T], T]
    stack_fn: Callable[[List[T], int], T]
    concat_fn: Callable[[List[T], int], T]
    matmul_fn: Callable[[T, T], T]
    clip_fn: Callable[[T, T, T], T]
    floor_fn: Callable[[T], T]
    ceil_fn: Callable[[T], T]
    gather_image_fn: Callable[[T, T], T]
    copy_fn: Callable[[T], T]
    max_fn: Callable[[T, int], T]
    min_fn: Callable[[T, int], T]
    logical_and_fn: Callable[[T, T], T]
    boolean_mask_fn: Callable[[T, T], T]


def _affine_transform(
        image: T,
        bboxes: T,
        labels: T,
        image_dest_tran_mat: T,
        bboxes_tran_mat: T,
        interpolation: Interpolation,
        d: _AffineDependency
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    """
    image, bboxes, labels = _sanitise(
        image, bboxes, labels, d.convert_fn, d.shape_fn, d.reshape_fn
    )

    image_shape = d.shape_fn(image)
    height, width = int(image_shape[0]), int(image_shape[1])
    h_mod, w_mod = height % 2, width % 2
    num_channels = int(image_shape[2])

    # Pad image to provide a zero-value pixel frame for clipping use below.
    pad_offsets = d.convert_fn([1, 1, 1, 1])
    image = d.pad_image_fn(image, pad_offsets)

    # References:
    # https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96

    # Destination indices. Note that (-foo // 2) != -(foo // 2).
    row_idxes = d.repeat_fn(  # Along y-axis, from top to bottom.
        d.range_fn(-(height // 2) + 1 - h_mod, height // 2 + 1, 1),
        d.round_to_int_fn(d.convert_fn([width]))
    )
    col_idxes = d.tile_fn(  # Along x-axis, from left to right.
        d.range_fn(-(width // 2) + 1 - w_mod, width // 2 + 1, 1),
        d.round_to_int_fn(d.convert_fn([height]))
    )
    # Note the (col, row) -> (x, y) swapping. Last axis is dummy.
    image_dest_idxes = d.stack_fn(
        [col_idxes, row_idxes, d.ones_like_fn(col_idxes)], 0
    )

    # Transform destination indices, with clipping. Note that these are floats.
    new_image_dest_idxes = d.matmul_fn(
        image_dest_tran_mat, d.convert_fn(image_dest_idxes)
    )
    clipped_new_image_dest_idxes = d.clip_fn(
        new_image_dest_idxes[:2],
        # Note the extra idx for the padded frame.
        d.convert_fn([
            [-(width // 2) - w_mod], [-(height // 2) - h_mod]
        ]),
        d.convert_fn([
            [width // 2 + 1], [height // 2 + 1]
        ])
    )

    # Assigning original pixel values to new positions.
    image_orig_idxes = d.concat_fn([
        # Rows.
        clipped_new_image_dest_idxes[1:] + d.convert_fn(height // 2 + h_mod),
        # Columns.
        clipped_new_image_dest_idxes[:1] + d.convert_fn(width // 2 + w_mod)
    ], 0)

    if interpolation == Interpolation.NEAREST:
        image_rounded_orig_idxes = d.round_to_int_fn(image_orig_idxes)
        values = d.gather_image_fn(image, image_rounded_orig_idxes)

    elif interpolation == Interpolation.BILINEAR:
        floor_floor_idxes = d.cast_to_int_fn(d.floor_fn(image_orig_idxes))
        ceil_ceil_idxes = d.cast_to_int_fn(d.ceil_fn(image_orig_idxes))

        floor_ceil_idxes = d.concat_fn(
            [floor_floor_idxes[:1, :], ceil_ceil_idxes[1:, :]], 0
        )
        ceil_floor_idxes = d.concat_fn(
            [ceil_ceil_idxes[:1, :], floor_floor_idxes[1:, :]], 0
        )

        dists: T = image_orig_idxes - d.convert_fn(floor_floor_idxes)
        # Reshape needed for broadcasting in the gather step.
        floor_weights = d.reshape_fn(1.0 - dists, (-1, 2))
        ceil_weights: T = 1.0 - floor_weights
        floor_floor_weights = floor_weights[:, :1] * floor_weights[:, 1:]
        floor_ceil_weights = floor_weights[:, :1] * ceil_weights[:, 1:]
        ceil_floor_weights = ceil_weights[:, :1] * floor_weights[:, 1:]
        ceil_ceil_weights = ceil_weights[:, :1] * ceil_weights[:, 1:]

        values = \
            d.gather_image_fn(image, floor_floor_idxes) * floor_floor_weights + \
            d.gather_image_fn(image, floor_ceil_idxes) * floor_ceil_weights + \
            d.gather_image_fn(image, ceil_floor_idxes) * ceil_floor_weights + \
            d.gather_image_fn(image, ceil_ceil_idxes) * ceil_ceil_weights

    else:
        raise ValueError("Undefined interpolation option.")

    new_image = d.reshape_fn(values, (height, width, num_channels))

    # Transform bboxes.
    top_left_xs = bboxes[:, :1]
    top_left_ys = bboxes[:, 1:2]
    top_right_xs = bboxes[:, :1] + bboxes[:, 2:3] - 1
    top_right_ys = bboxes[:, 1:2]
    bottom_left_xs = d.copy_fn(top_left_xs)
    bottom_left_ys = d.copy_fn(top_left_ys + bboxes[:, 3:] - 1)
    bottom_right_xs = d.copy_fn(top_right_xs)
    bottom_right_ys = d.copy_fn(top_right_ys + bboxes[:, 3:] - 1)

    xs = d.concat_fn(
        [top_left_xs - d.convert_fn(width // 2 - 1 + w_mod),
         top_right_xs - d.convert_fn(width // 2 - 1 + w_mod),
         bottom_left_xs - d.convert_fn(width // 2 - 1 + w_mod),
         bottom_right_xs - d.convert_fn(width // 2 - 1 + w_mod)],
        1
    )
    ys = d.concat_fn(
        [top_left_ys - d.convert_fn(height // 2 - 1 + h_mod),
         top_right_ys - d.convert_fn(height // 2 - 1 + h_mod),
         bottom_left_ys - d.convert_fn(height // 2 - 1 + h_mod),
         bottom_right_ys - d.convert_fn(height // 2 - 1 + h_mod)],
        1
    )
    bboxes_idxes = d.stack_fn(  # Shape: [num_bboxes, 3, 4].
        [xs, ys, d.ones_like_fn(xs)], 1
    )

    tran_bboxes_idxes = d.matmul_fn(bboxes_tran_mat, bboxes_idxes)

    # New bboxes, defined as the rectangle enclosing the transformed bboxes.
    tran_xs = tran_bboxes_idxes[:, 0, :]  # Shape: [num_bboxes, 4].
    tran_ys = tran_bboxes_idxes[:, 1, :]
    max_xs = d.max_fn(tran_xs, -1)  # Shape: [num_bboxes].
    max_ys = d.max_fn(tran_ys, -1)
    min_xs = d.min_fn(tran_xs, -1)
    min_ys = d.min_fn(tran_ys, -1)

    one = d.cast_to_int_fn(d.convert_fn(1))

    tran_top_left_xs = d.round_to_int_fn(d.expand_dim_fn(min_xs, -1))
    tran_top_left_ys = d.round_to_int_fn(d.expand_dim_fn(min_ys, -1))
    tran_bottom_right_xs = d.round_to_int_fn(d.expand_dim_fn(max_xs, -1))
    tran_bottom_right_ys = d.round_to_int_fn(d.expand_dim_fn(max_ys, -1))
    new_widths: T = tran_bottom_right_xs - tran_top_left_xs + one
    new_heights: T = tran_bottom_right_ys - tran_top_left_ys + one
    tran_bboxes = d.concat_fn([  # Shape: [num_bboxes, 4].
        tran_top_left_xs, tran_top_left_ys, new_widths, new_heights
    ], -1)

    new_xs = tran_bboxes[:, :1] + width // 2 + w_mod - 1
    new_ys = tran_bboxes[:, 1:2] + height // 2 + h_mod - 1

    # Filter new bboxes values.
    xcens = new_xs + new_widths // 2
    ycens = new_ys + new_heights // 2
    included = d.squeeze_fn(d.logical_and_fn(
        d.logical_and_fn(xcens >= 0, xcens <= width),
        d.logical_and_fn(ycens >= 0, ycens <= height)
    ), -1)

    # Clip new bboxes values.
    xmaxs = d.clip_fn(
        d.convert_fn(new_xs + new_widths), d.convert_fn(0), d.convert_fn(width)
    )
    ymaxs = d.clip_fn(
        d.convert_fn(new_ys + new_heights), d.convert_fn(0), d.convert_fn(height)
    )
    new_xs = d.clip_fn(
        d.convert_fn(new_xs), d.convert_fn(0), d.convert_fn(width)
    )
    new_ys = d.clip_fn(
        d.convert_fn(new_ys), d.convert_fn(0), d.convert_fn(height)
    )
    new_widths = xmaxs - new_xs
    new_heights = ymaxs - new_ys

    new_bboxes = d.concat_fn([new_xs, new_ys, new_widths, new_heights], 1)
    new_bboxes = d.convert_fn(d.boolean_mask_fn(new_bboxes, included))

    # Filter labels.
    new_labels = d.boolean_mask_fn(labels, included)

    return new_image, new_bboxes, new_labels


def _get_flip_left_right_mats(
        convert_fn: Callable[..., T]
) -> Tuple[T, T]:
    image_dest_flip_lr_mat = convert_fn([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    bboxes_flip_lr_mat = convert_fn([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    return image_dest_flip_lr_mat, bboxes_flip_lr_mat


def _flip_left_right(
        image: T,
        bboxes: T,
        labels: T,
        interpolation: Interpolation,
        d: _AffineDependency
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    """
    image_dest_flip_lr_mat, bboxes_flip_lr_mat = _get_flip_left_right_mats(
        d.convert_fn
    )
    return _affine_transform(
        image, bboxes, labels, image_dest_flip_lr_mat, bboxes_flip_lr_mat,
        interpolation, d
    )


def _get_flip_up_down_mats(
        convert_fn: Callable[..., T]
) -> Tuple[T, T]:
    image_dest_flip_ud_mat = convert_fn([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    bboxes_flip_ud_mat = convert_fn([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    return image_dest_flip_ud_mat, bboxes_flip_ud_mat


def _flip_up_down(
        image: T,
        bboxes: T,
        labels: T,
        interpolation: Interpolation,
        d: _AffineDependency
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    """
    image_dest_flip_ud_mat, bboxes_flip_ud_mat = _get_flip_up_down_mats(
        d.convert_fn
    )
    return _affine_transform(
        image, bboxes, labels, image_dest_flip_ud_mat, bboxes_flip_ud_mat,
        interpolation, d
    )


def _get_rotate_mats(
        angle_deg: T,
        convert_fn: Callable[..., T],
        cos_fn: Callable[[T], T],
        sin_fn: Callable[[T], T]
) -> Tuple[T, T]:
    ang_rad = convert_fn(np.pi * angle_deg / 180.0)
    # Image rotation matrix. Clockwise for the destination indices,
    # so the final image would appear to be rotated anti-clockwise.
    image_dest_rot_mat = convert_fn([
        [cos_fn(ang_rad), -sin_fn(ang_rad), 0],
        [sin_fn(ang_rad), cos_fn(ang_rad), 0],
        [0, 0, 1]
    ])
    bboxes_rot_mat = convert_fn([  # Anti-clockwise.
        [cos_fn(ang_rad), sin_fn(ang_rad), 0],
        [-sin_fn(ang_rad), cos_fn(ang_rad), 0],
        [0, 0, 1]
    ])
    return image_dest_rot_mat, bboxes_rot_mat


def _rotate(
        image: T,
        bboxes: T,
        labels: T,
        angle_deg: T,
        cos_fn: Callable[[T], T],
        sin_fn: Callable[[T], T],
        interpolation: Interpolation,
        d: _AffineDependency
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    angle_deg: positive means anti-clockwise.
    """
    image_dest_rot_mat, bboxes_rot_mat = _get_rotate_mats(
        angle_deg, d.convert_fn, cos_fn, sin_fn
    )
    return _affine_transform(
        image, bboxes, labels, image_dest_rot_mat, bboxes_rot_mat,
        interpolation, d
    )


def _get_shear_mats(
        angle_deg: T,
        convert_fn: Callable[..., T],
        tan_fn: Callable[[T], T]
) -> Tuple[T, T]:
    ang_rad = convert_fn(np.pi * angle_deg / 180.0)
    factor = tan_fn(ang_rad)
    # Image shear matrix. Clockwise for the destination indices,
    # so the final image would appear to be sheared anti-clockwise.
    image_dest_shear_mat = convert_fn([
        [1, -factor, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    bboxes_shear_mat = convert_fn([  # Anti-clockwise.
        [1, factor, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    return image_dest_shear_mat, bboxes_shear_mat


def _shear(
        image: T,
        bboxes: T,
        labels: T,
        angle_deg: T,
        tan_fn: Callable[[T], T],
        interpolation: Interpolation,
        d: _AffineDependency
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    angle_deg: positive means anti-clockwise, where abs(angle_deg) must be < 90.
    """
    image_dest_shear_mat, bboxes_shear_mat = _get_shear_mats(
        angle_deg, d.convert_fn, tan_fn
    )
    return _affine_transform(
        image, bboxes, labels, image_dest_shear_mat, bboxes_shear_mat,
        interpolation, d
    )


def _get_translate_mats(
        translate_height: T,
        translate_width: T,
        convert_fn: Callable[..., T]
) -> Tuple[T, T]:
    image_dest_translate_mat = convert_fn([
        [1, 0, -translate_width],
        [0, 1, -translate_height],
        [0, 0, 1]
    ])
    bboxes_translate_mat = convert_fn([
        [1, 0, translate_width],
        [0, 1, translate_height],
        [0, 0, 1]
    ])
    return image_dest_translate_mat, bboxes_translate_mat


def _translate(
        image: T,
        bboxes: T,
        labels: T,
        translate_height: T,
        translate_width: T,
        interpolation: Interpolation,
        d: _AffineDependency
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    translate_height: in range (-image_height, image_height)
    translate_width: in range (-image_width, image_width)
    """
    image_dest_translate_mat, bboxes_translate_mat = _get_translate_mats(
        translate_height, translate_width, d.convert_fn
    )
    return _affine_transform(
        image, bboxes, labels, image_dest_translate_mat, bboxes_translate_mat,
        interpolation, d
    )


def _get_random_size_fractions(
        height_fraction_range_: Tuple[float, float],
        width_fraction_range_: Tuple[float, float],
        rand_fn: Callable[..., T],
        convert_fn: Callable[..., T],
) -> Tuple[T, T]:
    """
    height_fraction_range, width_fraction_range: (-1.0, 1.0)
    rand_fn: generate random number in range [0.0, 1.0)
    """
    height_fraction_range = convert_fn(height_fraction_range_)
    width_fraction_range = convert_fn(width_fraction_range_)

    min_height_fraction = height_fraction_range[0]
    min_width_fraction = width_fraction_range[0]
    height_fraction_diff = height_fraction_range[1] - min_height_fraction
    width_fraction_diff = width_fraction_range[1] - min_width_fraction

    height_fraction = height_fraction_diff * rand_fn() + min_height_fraction
    width_fraction = width_fraction_diff * rand_fn() + min_width_fraction

    return height_fraction, width_fraction


def _get_crop_inputs(
        image_height_: int,
        image_width_: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., T],
        convert_fn: Callable[..., T],
        round_to_int_fn: Callable[[T], T]
) -> Tuple[T, T, T, T]:
    """
    height_fraction_range, width_fraction_range: in range [0.0, 1.0)
    rand_fn: generate random number in range [0.0, 1.0)
    """
    image_height = convert_fn(image_height_)
    image_width = convert_fn(image_width_)

    height_fraction, width_fraction = _get_random_size_fractions(
        height_fraction_range, width_fraction_range, rand_fn, convert_fn
    )

    crop_height: T = image_height * height_fraction
    crop_width: T = image_width * width_fraction

    offset_height = round_to_int_fn((image_height - crop_height) * rand_fn())
    offset_width = round_to_int_fn((image_width - crop_width) * rand_fn())

    return offset_height, offset_width, crop_height, crop_width


def _crop(
        image: T,
        bboxes: T,
        labels: T,
        offset_height: T,
        offset_width: T,
        crop_height: T,
        crop_width: T,
        convert_fn: Callable[..., T],
        shape_fn: Callable[[T], Sequence[int]],
        reshape_fn: Callable[[T, Sequence[int]], T],
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
    offset_height: in range [0, image_height)
    offset_width: in range [0, image_width)
    """
    image, bboxes, labels = _sanitise(
        image, bboxes, labels, convert_fn, shape_fn, reshape_fn
    )

    top = offset_height
    left = offset_width
    bottom: T = top + crop_height
    right: T = left + crop_width

    # Crop image.
    image = image[int(top):int(bottom), int(left):int(right), :]

    # Translate bboxes values.
    xs: T = bboxes[:, :1] - offset_width
    ys: T = bboxes[:, 1:2] - offset_height
    widths = bboxes[:, 2:3]
    heights = bboxes[:, 3:]

    # Filter bboxes values.
    xcens = xs + widths // 2
    ycens = ys + heights // 2
    included = squeeze_fn(logical_and_fn(
        logical_and_fn(xcens >= 0, xcens <= crop_width),
        logical_and_fn(ycens >= 0, ycens <= crop_height)
    ), -1)  # Squeeze along the last axis.

    # Clip bboxes values.
    xmaxs = clip_fn(xs + widths, convert_fn(0), crop_width)
    ymaxs = clip_fn(ys + heights, convert_fn(0), crop_height)
    xs = clip_fn(xs, convert_fn(0), crop_width)
    ys = clip_fn(ys, convert_fn(0), crop_height)
    widths = xmaxs - xs
    heights = ymaxs - ys

    bboxes = concat_fn([xs, ys, widths, heights], 1)
    bboxes = boolean_mask_fn(bboxes, included)

    # Filter labels.
    labels = boolean_mask_fn(labels, included)

    return image, bboxes, labels


def _resize(
        image: T,
        bboxes: T,
        labels: T,
        dest_size: Tuple[int, int],
        convert_fn: Callable[..., T],
        shape_fn: Callable[[T], Sequence[int]],
        reshape_fn: Callable[[T, Sequence[int]], T],
        resize_image_fn: Callable[[T, Tuple[int, int]], T],
        concat_fn: Callable[[List[T], int], T],
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    dest_size: (image_height, image_width)
    """
    image, bboxes, labels = _sanitise(
        image, bboxes, labels, convert_fn, shape_fn, reshape_fn
    )

    image_shape = shape_fn(image)
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
