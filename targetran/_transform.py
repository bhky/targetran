"""
Image and target transform utilities.
"""

from typing import Any, Callable, List, Tuple, TypeVar

import numpy as np  # type: ignore

# This roughly means anything that is ndarray-like.
T = TypeVar("T", np.ndarray, Any)


def _sanitise(
        image: T,
        bboxes: T,
        labels: T,
        convert_fn: Callable[..., T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
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


def _affine_transform(
        image: T,
        bboxes: T,
        labels: T,
        convert_fn: Callable[..., T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        pad_image_fn: Callable[[T, T], T],
        range_fn: Callable[[int, int, int], T],
        round_to_int_fn: Callable[[T], T],
        repeat_fn: Callable[[T, T], T],
        tile_fn: Callable[[T, T], T],
        ones_like_fn: Callable[[T], T],
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
    image, bboxes, labels = _sanitise(
        image, bboxes, labels, convert_fn, shape_fn, reshape_fn
    )

    image_shape = shape_fn(image)
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
    # Note the (col, row) -> (x, y) swapping. Last axis is dummy.
    image_dest_idxes = stack_fn(
        [col_idxes, row_idxes, ones_like_fn(col_idxes)], 0
    )

    # Transform image, with clipping.
    new_image_dest_idxes = matmul_fn(
        image_dest_tran_mat, convert_fn(image_dest_idxes)
    )
    clipped_new_image_dest_idxes = clip_fn(
        new_image_dest_idxes[:2],
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
    bboxes_idxes = stack_fn(  # Shape: [num_bboxes, 3, 4].
        [xs, ys, ones_like_fn(xs)], 1
    )

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
        convert_fn: Callable[..., T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        pad_image_fn: Callable[[T, T], T],
        range_fn: Callable[[int, int, int], T],
        round_to_int_fn: Callable[[T], T],
        repeat_fn: Callable[[T, T], T],
        tile_fn: Callable[[T, T], T],
        ones_like_fn: Callable[[T], T],
        stack_fn: Callable[[List[T], int], T],
        concat_fn: Callable[[List[T], int], T],
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
    image_dest_flip_lr_mat, bboxes_flip_lr_mat = _get_flip_left_right_mats(
        convert_fn
    )
    return _affine_transform(
        image, bboxes, labels, convert_fn, shape_fn, reshape_fn,
        expand_dim_fn, squeeze_fn, pad_image_fn, range_fn, round_to_int_fn,
        repeat_fn, tile_fn, ones_like_fn, stack_fn, concat_fn,
        image_dest_flip_lr_mat, bboxes_flip_lr_mat, matmul_fn,
        clip_fn, gather_image_fn, copy_fn, max_fn, min_fn,
        logical_and_fn, boolean_mask_fn
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
        convert_fn: Callable[..., T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        pad_image_fn: Callable[[T, T], T],
        range_fn: Callable[[int, int, int], T],
        round_to_int_fn: Callable[[T], T],
        repeat_fn: Callable[[T, T], T],
        tile_fn: Callable[[T, T], T],
        ones_like_fn: Callable[[T], T],
        stack_fn: Callable[[List[T], int], T],
        concat_fn: Callable[[List[T], int], T],
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
    image_dest_flip_ud_mat, bboxes_flip_ud_mat = _get_flip_up_down_mats(
        convert_fn
    )
    return _affine_transform(
        image, bboxes, labels, convert_fn, shape_fn, reshape_fn,
        expand_dim_fn, squeeze_fn, pad_image_fn, range_fn, round_to_int_fn,
        repeat_fn, tile_fn, ones_like_fn, stack_fn, concat_fn,
        image_dest_flip_ud_mat, bboxes_flip_ud_mat, matmul_fn,
        clip_fn, gather_image_fn, copy_fn, max_fn, min_fn,
        logical_and_fn, boolean_mask_fn
    )


def _get_rotate_mats(
        angle_deg: T,
        convert_fn: Callable[..., T],
        cos_fn: Callable[[T], T],
        sin_fn: Callable[[T], T]
) -> Tuple[T, T]:
    ang_rad = np.pi * angle_deg / 180.0
    # Image rotation matrix. Clockwise for the destination indices,
    # so the final image would appear to be rotated anti-clockwise.
    image_dest_rot_mat = convert_fn([
        [cos_fn(ang_rad), -sin_fn(ang_rad), convert_fn(0)],
        [sin_fn(ang_rad), cos_fn(ang_rad), convert_fn(0)],
        [convert_fn(0), convert_fn(0), convert_fn(1)]
    ])
    bboxes_rot_mat = convert_fn([  # Anti-clockwise.
        [cos_fn(ang_rad), sin_fn(ang_rad), convert_fn(0)],
        [-sin_fn(ang_rad), cos_fn(ang_rad), convert_fn(0)],
        [convert_fn(0), convert_fn(0), convert_fn(1)]
    ])
    return image_dest_rot_mat, bboxes_rot_mat


def _rotate(
        image: T,
        bboxes: T,
        labels: T,
        angle_deg: T,
        convert_fn: Callable[..., T],
        cos_fn: Callable[[T], T],
        sin_fn: Callable[[T], T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        pad_image_fn: Callable[[T, T], T],
        range_fn: Callable[[int, int, int], T],
        round_to_int_fn: Callable[[T], T],
        repeat_fn: Callable[[T, T], T],
        tile_fn: Callable[[T, T], T],
        ones_like_fn: Callable[[T], T],
        stack_fn: Callable[[List[T], int], T],
        concat_fn: Callable[[List[T], int], T],
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
    angle_deg: positive means anti-clockwise.
    """
    image_dest_rot_mat, bboxes_rot_mat = _get_rotate_mats(
        angle_deg, convert_fn, cos_fn, sin_fn
    )
    return _affine_transform(
        image, bboxes, labels, convert_fn, shape_fn, reshape_fn,
        expand_dim_fn, squeeze_fn, pad_image_fn, range_fn, round_to_int_fn,
        repeat_fn, tile_fn, ones_like_fn, stack_fn, concat_fn,
        image_dest_rot_mat, bboxes_rot_mat, matmul_fn,
        clip_fn, gather_image_fn, copy_fn, max_fn, min_fn,
        logical_and_fn, boolean_mask_fn
    )


def _get_shear_mats(
        angle_deg: T,
        convert_fn: Callable[..., T],
        tan_fn: Callable[[T], T]
) -> Tuple[T, T]:
    ang_rad = np.pi * angle_deg / 180.0
    factor = tan_fn(ang_rad)
    # Image shear matrix. Clockwise for the destination indices,
    # so the final image would appear to be sheared anti-clockwise.
    image_dest_shear_mat = convert_fn([
        [convert_fn(1), -factor, convert_fn(0)],
        [convert_fn(0), convert_fn(1), convert_fn(0)],
        [convert_fn(0), convert_fn(0), convert_fn(1)]
    ])
    bboxes_shear_mat = convert_fn([  # Anti-clockwise.
        [convert_fn(1), factor, convert_fn(0)],
        [convert_fn(0), convert_fn(1), convert_fn(0)],
        [convert_fn(0), convert_fn(0), convert_fn(1)]
    ])
    return image_dest_shear_mat, bboxes_shear_mat


def _shear(
        image: T,
        bboxes: T,
        labels: T,
        angle_deg: T,
        convert_fn: Callable[..., T],
        tan_fn: Callable[[T], T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        pad_image_fn: Callable[[T, T], T],
        range_fn: Callable[[int, int, int], T],
        round_to_int_fn: Callable[[T], T],
        repeat_fn: Callable[[T, T], T],
        tile_fn: Callable[[T, T], T],
        ones_like_fn: Callable[[T], T],
        stack_fn: Callable[[List[T], int], T],
        concat_fn: Callable[[List[T], int], T],
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
    angle_deg: positive means anti-clockwise, where abs(angle_deg) must be < 90.
    """
    image_dest_shear_mat, bboxes_shear_mat = _get_shear_mats(
        angle_deg, convert_fn, tan_fn
    )
    return _affine_transform(
        image, bboxes, labels, convert_fn, shape_fn, reshape_fn,
        expand_dim_fn, squeeze_fn, pad_image_fn, range_fn, round_to_int_fn,
        repeat_fn, tile_fn, ones_like_fn, stack_fn, concat_fn,
        image_dest_shear_mat, bboxes_shear_mat, matmul_fn,
        clip_fn, gather_image_fn, copy_fn, max_fn, min_fn,
        logical_and_fn, boolean_mask_fn
    )


def _get_translate_mats(
        translate_height: T,
        translate_width: T,
        convert_fn: Callable[..., T]
) -> Tuple[T, T]:
    image_dest_translate_mat = convert_fn([
        [convert_fn(1), convert_fn(0), -convert_fn(translate_width)],
        [convert_fn(0), convert_fn(1), -convert_fn(translate_height)],
        [convert_fn(0), convert_fn(0), convert_fn(1)]
    ])
    bboxes_translate_mat = convert_fn([
        [convert_fn(1), convert_fn(0), convert_fn(translate_width)],
        [convert_fn(0), convert_fn(1), convert_fn(translate_height)],
        [convert_fn(0), convert_fn(0), convert_fn(1)]
    ])
    return image_dest_translate_mat, bboxes_translate_mat


def _translate(
        image: T,
        bboxes: T,
        labels: T,
        translate_height: T,
        translate_width: T,
        convert_fn: Callable[..., T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        expand_dim_fn: Callable[[T, int], T],
        squeeze_fn: Callable[[T, int], T],
        pad_image_fn: Callable[[T, T], T],
        range_fn: Callable[[int, int, int], T],
        round_to_int_fn: Callable[[T], T],
        repeat_fn: Callable[[T, T], T],
        tile_fn: Callable[[T, T], T],
        ones_like_fn: Callable[[T], T],
        stack_fn: Callable[[List[T], int], T],
        concat_fn: Callable[[List[T], int], T],
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
    translate_height: in range (-image_height, image_height)
    translate_width: in range (-image_width, image_width)
    """
    image_dest_translate_mat, bboxes_translate_mat = _get_translate_mats(
        translate_height, translate_width, convert_fn
    )
    return _affine_transform(
        image, bboxes, labels, convert_fn, shape_fn, reshape_fn,
        expand_dim_fn, squeeze_fn, pad_image_fn, range_fn, round_to_int_fn,
        repeat_fn, tile_fn, ones_like_fn, stack_fn, concat_fn,
        image_dest_translate_mat, bboxes_translate_mat, matmul_fn,
        clip_fn, gather_image_fn, copy_fn, max_fn, min_fn,
        logical_and_fn, boolean_mask_fn
    )


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


def _get_crop_inputs(
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
    """
    image_height = convert_fn(image_height)
    image_width = convert_fn(image_width)

    height_fractions, width_fractions = _get_random_size_fractions(
        height_fraction_range, width_fraction_range, rand_fn, convert_fn
    )

    cropped_image_heights = image_height * height_fractions
    cropped_image_widths = image_width * width_fractions

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
        offset_height: T,
        offset_width: T,
        cropped_image_height: T,
        cropped_image_width: T,
        convert_fn: Callable[..., T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
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


def _resize(
        image: T,
        bboxes: T,
        labels: T,
        dest_size: Tuple[int, int],
        convert_fn: Callable[..., T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, ...]], T],
        resize_image_fn: Callable[[T, Tuple[int, int]], T],
        concat_fn: Callable[[List[T], int], T],
) -> Tuple[T, T, T]:
    """
    image: [h, w, c]
    bboxes: [[top_left_x, top_left_y, width, height], ...]
    labels: [0, 1, 0, ...]
    dest_size: (height, width)
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
