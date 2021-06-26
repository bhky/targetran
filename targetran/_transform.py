"""
Image and target transform utilities.
"""

from typing import Any, Callable, List, Tuple, TypeVar

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

from ._functional import _np_convert, _np_resize_images, _np_boolean_mask
from ._functional import _np_multiply, _np_logical_and, _np_make_bboxes_list
from ._functional import _tf_convert, _tf_resize_images, _tf_make_bboxes_list


T = TypeVar("T", np.ndarray, tf.Tensor)


def _reshape_bboxes(
        bboxes_list: List[T],
        reshape_fn: Callable[[T, Tuple[int, int]], T]
) -> List[T]:
    """
    This seemingly extra process is mainly for tackling empty bboxes array.
    """
    return [reshape_fn(bboxes, (-1, 4)) for bboxes in bboxes_list]


def _flip_left_right(
        images: T,
        bboxes_list: List[T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, int]], T],
        convert_fn: Callable[..., T],
        concat_fn: Callable[[List[T], int], T],
        make_bboxes_list_fn: Callable[[T, List[int]], List[T]]
) -> Tuple[T, List[T]]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    image_width = convert_fn(images_shape[2])

    images = images[..., ::-1, :]

    bboxes_list = _reshape_bboxes(bboxes_list, reshape_fn)
    all_bboxes = concat_fn(bboxes_list, 0)  # Along axis 0.
    assert shape_fn(all_bboxes)[-1] == 4

    xs = image_width - all_bboxes[:, :1] - all_bboxes[:, 2:3]
    all_bboxes = concat_fn([xs, all_bboxes[:, 1:]], 1)  # Along axis 1.

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    bboxes_list = make_bboxes_list_fn(all_bboxes, bboxes_nums)

    return images, bboxes_list


def _flip_up_down(
        images: T,
        bboxes_list: List[T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, int]], T],
        convert_fn: Callable[..., T],
        concat_fn: Callable[[List[T], int], T],
        make_bboxes_list_fn: Callable[[T, List[int]], List[T]]
) -> Tuple[T, List[T]]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    image_height = convert_fn(images_shape[1])

    images = images[:, ::-1, ...]

    bboxes_list = _reshape_bboxes(bboxes_list, reshape_fn)
    all_bboxes = concat_fn(bboxes_list, 0)  # Along axis 0.
    assert shape_fn(all_bboxes)[-1] == 4

    ys = image_height - all_bboxes[:, 1:2] - all_bboxes[:, 3:]
    all_bboxes = concat_fn(
        [all_bboxes[:, :1], ys, all_bboxes[:, 2:]], 1  # Along axis 1.
    )

    bboxes_nums = [len(bboxes) for bboxes in bboxes_list]
    bboxes_list = make_bboxes_list_fn(all_bboxes, bboxes_nums)

    return images, bboxes_list


def _rotate_90(
        images: T,
        bboxes_list: List[T],
        shape_fn: Callable[[T], Tuple[int, ...]],
        reshape_fn: Callable[[T, Tuple[int, int]], T],
        convert_fn: Callable[..., T],
        transpose_fn: Callable[[T, Tuple[int, ...]], T],
        concat_fn: Callable[[List[T], int], T],
        make_bboxes_list_fn: Callable[[T, List[int]], List[T]]
) -> Tuple[T, List[T]]:
    """
    Rotate 90 degrees anti-clockwise.
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    image_width = convert_fn(images_shape[2])

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
    bboxes_list = make_bboxes_list_fn(all_bboxes, bboxes_nums)

    return images, bboxes_list


def _resize(
        images: T,
        bboxes_list: List[T],
        dest_size: Tuple[int, int],
        shape_fn: Callable[[T], Tuple[int, ...]],
        resize_images_fn: Callable[[T, Tuple[int, int]], T],
        convert_fn: Callable[..., T],
        concat_fn: Callable[[List[T], int], T],
) -> Tuple[T, List[T]]:
    """
    images: [bs, h, w, c]
    bboxes (for one image): [[top_left_x, top_left_y, width, height], ...]
    dest_size: (height, width)
    """
    images_shape = shape_fn(images)
    assert len(images_shape) == 4

    images = resize_images_fn(images, dest_size)

    w = convert_fn(dest_size[1] / images_shape[2])
    h = convert_fn(dest_size[0] / images_shape[1])

    def resize_bboxes(bboxes: T) -> T:
        xs = bboxes[:, :1] * w
        ys = bboxes[:, 1:2] * h
        widths = bboxes[:, 2:3] * w
        heights = bboxes[:, 3:] * h
        return concat_fn([xs, ys, widths, heights], 1)

    bboxes_list = [resize_bboxes(bboxes) for bboxes in bboxes_list]

    return images, bboxes_list


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
        bboxes_list: List[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    return _flip_left_right(
        images, bboxes_list,
        np.shape, np.reshape, _np_convert, np.concatenate, _np_make_bboxes_list
    )


def _np_flip_up_down(
        images: np.ndarray,
        bboxes_list: List[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    return _flip_up_down(
        images, bboxes_list,
        np.shape, np.reshape, _np_convert, np.concatenate, _np_make_bboxes_list
    )


def _np_rotate_90(
        images: np.ndarray,
        bboxes_list: List[np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray]]:
    return _rotate_90(
        images, bboxes_list,
        np.shape, np.reshape, _np_convert, np.transpose,
        np.concatenate, _np_make_bboxes_list
    )


def _np_crop_and_resize(
        images: np.ndarray,
        bboxes_list: List[np.ndarray],
        x_offset_fractions: np.ndarray,
        y_offset_fractions: np.ndarray
) -> Tuple[np.ndarray, List[np.ndarray]]:

    images_shape = np.shape(images)
    assert len(images_shape) == 4

    tuples = [
        _crop_single(
            image, bboxes,
            x_offset_fraction, y_offset_fraction,
            np.shape, np.reshape, _np_convert,
            _np_multiply, np.rint, np.abs, np.where, np.concatenate,
            _np_logical_and, np.squeeze, _np_boolean_mask
        ) for image, bboxes, x_offset_fraction, y_offset_fraction in zip(
            images, bboxes_list, x_offset_fractions, y_offset_fractions
        )
    ]
    image_list, bboxes_list = zip(*tuples)
    images = np.array(image_list)
    return _resize(
        images, bboxes_list,
        images_shape[1:3], np.shape, _np_resize_images,
        _np_convert, np.concatenate
    )


def _tf_flip_left_right(
        images: tf.Tensor,
        bboxes_list: List[tf.Tensor]
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    return _flip_left_right(
        images, bboxes_list,
        tf.shape, tf.reshape, _tf_convert, tf.concat, _tf_make_bboxes_list
    )


def _tf_flip_up_down(
        images: tf.Tensor,
        bboxes_list: List[tf.Tensor]
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    return _flip_up_down(
        images, bboxes_list,
        tf.shape, tf.reshape, _tf_convert, tf.concat, _tf_make_bboxes_list
    )


def _tf_rotate_90(
        images: tf.Tensor,
        bboxes_list: List[tf.Tensor]
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    return _rotate_90(
        images, bboxes_list,
        tf.shape, tf.reshape, _tf_convert, tf.transpose, tf.concat,
        _tf_make_bboxes_list
    )


def _tf_crop_and_resize(
        images: tf.Tensor,
        bboxes_list: List[tf.Tensor],
        x_offset_fractions: tf.Tensor,
        y_offset_fractions: tf.Tensor
) -> Tuple[tf.Tensor, List[tf.Tensor]]:

    images_shape = np.shape(images)
    assert len(images_shape) == 4

    tuples = [
        _crop_single(
            image, bboxes,
            x_offset_fraction, y_offset_fraction,
            tf.shape, tf.reshape, _tf_convert,
            tf.multiply, tf.math.rint, tf.abs, tf.where, tf.concat,
            tf.logical_and, tf.squeeze, tf.boolean_mask
        ) for image, bboxes, x_offset_fraction, y_offset_fraction in zip(
            images, bboxes_list, x_offset_fractions, y_offset_fractions
        )
    ]
    image_list, bboxes_list = zip(*tuples)
    images = tf.convert_to_tensor(image_list)
    return _resize(
        images, bboxes_list,
        images_shape[1:3], tf.shape, _tf_resize_images,
        _tf_convert, tf.concat
    )


class _TFRandomBase:

    def __init__(
            self,
            tf_fn: Callable[..., Tuple[tf.Tensor, List[tf.Tensor]]],
            flip_probability: float,
            seed: int,
    ) -> None:
        self._tf_fn = tf_fn
        self.flip_probability = flip_probability
        self.seed = seed

    def call(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor],
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:

        rand = tf.random.uniform(shape=tf.shape(images)[:1], seed=self.seed)
        output: Tuple[tf.Tensor, List[tf.Tensor]] = tf.where(
            tf.less(rand, self.flip_probability),
            self._tf_fn(images, bboxes_list, *args, **kwargs),
            (images, bboxes_list)
        )
        return output


class TFRandomFlipLeftRight(_TFRandomBase):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_flip_left_right, flip_probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(images, bboxes_list)


class TFRandomFlipUpDown(_TFRandomBase):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_flip_up_down, flip_probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(images, bboxes_list)


class TFRandomRotate90(_TFRandomBase):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_rotate_90, flip_probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(images, bboxes_list)


class TFRandomCropAndResize(_TFRandomBase):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_crop_and_resize, flip_probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor],
            x_offset_fractions: tf.Tensor,
            y_offset_fractions: tf.Tensor
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(
            images,
            bboxes_list,
            x_offset_fractions,
            y_offset_fractions
        )
