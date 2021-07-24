"""
API for Numpy usage.
"""

from typing import Any, Callable, Tuple

import numpy as np  # type: ignore

from ._functional import (
    _np_map_idx_fn,
    _np_convert,
    _np_range,
    _np_stack_bboxes,
    _np_round_to_int,
    _np_resize_image,
    _np_boolean_mask,
    _np_logical_and,
    _np_pad_images,
    _np_gather_image,
    _np_make_bboxes_ragged
)

from ._transform import (
    _flip_left_right,
    _flip_up_down,
    _rotate_90,
    _rotate_90_and_pad,
    _rotate_single,
    _shear_single,
    _crop_single,
    _resize_single,
    _translate_single,
    _get_random_crop_inputs,
    _get_random_size_fractions
)


def flip_left_right(
        images: np.ndarray,
        bboxes_ragged: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return _flip_left_right(
        images, bboxes_ragged,
        np.shape, _np_convert, _np_stack_bboxes, np.concatenate,
        _np_make_bboxes_ragged
    )


def flip_up_down(
        images: np.ndarray,
        bboxes_ragged: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return _flip_up_down(
        images, bboxes_ragged,
        np.shape, _np_convert, _np_stack_bboxes, np.concatenate,
        _np_make_bboxes_ragged
    )


def _np_resize_single(
        image: np.ndarray,
        bboxes: np.ndarray,
        dest_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    return _resize_single(
        image, bboxes,
        dest_size, np.shape, _np_resize_image, _np_convert, np.concatenate
    )


def resize(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        dest_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:

    def fn(idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return _np_resize_single(images[idx], bboxes_ragged[idx], dest_size)

    return _np_map_idx_fn(fn, int(np.shape(images)[0]))


def rotate_90(
        images: np.ndarray,
        bboxes_ragged: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return _rotate_90(
        images, bboxes_ragged,
        np.shape, _np_convert, np.transpose, _np_stack_bboxes, np.concatenate,
        _np_make_bboxes_ragged
    )


def _np_rotate_90_and_pad(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Middle-step function for easy testing.
    """
    return _rotate_90_and_pad(
        images, bboxes_ragged,
        np.shape, _np_convert, np.transpose, _np_stack_bboxes, np.concatenate,
        np.where, np.ceil, np.floor, _np_pad_images,
        _np_make_bboxes_ragged
    )


def rotate_90_and_resize(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Could be rotate_90_and_pad_and_resize, but thought it is too clumsy.
    """
    height, width = int(np.shape(images)[1]), int(np.shape(images)[2])
    images, bboxes_ragged = _np_rotate_90_and_pad(images, bboxes_ragged)
    return resize(images, bboxes_ragged, (height, width))


def _np_rotate_single(
        image: np.ndarray,
        bboxes: np.ndarray,
        angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    return _rotate_single(
        image, bboxes, angle_deg,
        np.shape, _np_convert, np.expand_dims, np.squeeze,
        _np_pad_images, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.stack, np.concatenate, np.cos, np.sin, np.matmul, np.clip,
        _np_gather_image, np.reshape, np.copy,
        np.max, np.min, _np_logical_and, _np_boolean_mask
    )


def rotate(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        angles_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    def fn(idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return _np_rotate_single(
            images[idx], bboxes_ragged[idx], angles_deg[idx]
        )

    return _np_map_idx_fn(fn, int(np.shape(images)[0]))


def _np_shear_single(
        image: np.ndarray,
        bboxes: np.ndarray,
        angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    return _shear_single(
        image, bboxes, angle_deg,
        np.shape, _np_convert, np.expand_dims, np.squeeze,
        _np_pad_images, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.stack, np.concatenate, np.tan, np.matmul, np.clip,
        _np_gather_image, np.reshape, np.copy,
        np.max, np.min, _np_logical_and, _np_boolean_mask
    )


def shear(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        angles_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    def fn(idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return _np_shear_single(
            images[idx], bboxes_ragged[idx], angles_deg[idx],
        )

    return _np_map_idx_fn(fn, int(np.shape(images)[0]))


def _np_get_random_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _get_random_crop_inputs(
        image_height, image_width, height_fraction_range, width_fraction_range,
        rand_fn, _np_convert, _np_round_to_int
    )


def _np_crop_single(
        image: np.ndarray,
        bboxes: np.ndarray,
        offset_height: int,
        offset_width: int,
        cropped_image_height: int,
        cropped_image_width: int
) -> Tuple[np.ndarray, np.ndarray]:
    return _crop_single(
        image, bboxes,
        offset_height, offset_width,
        cropped_image_height, cropped_image_width,
        np.shape, np.reshape, _np_convert, np.concatenate,
        _np_logical_and, np.squeeze, _np_boolean_mask
    )


def crop_and_resize(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        offset_heights: np.ndarray,
        offset_widths: np.ndarray,
        cropped_image_heights: np.ndarray,
        cropped_image_widths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    def fn(idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image, bboxes = _np_crop_single(
            images[idx],
            bboxes_ragged[idx],
            offset_heights[idx], offset_widths[idx],
            cropped_image_heights[idx], cropped_image_widths[idx]
        )
        return _np_resize_single(
            image, bboxes, np.shape(images)[1:3]
        )

    return _np_map_idx_fn(fn, int(np.shape(images)[0]))


def _np_translate_single(
        image: np.ndarray,
        bboxes: np.ndarray,
        translate_height: int,
        translate_width: int
) -> Tuple[np.ndarray, np.ndarray]:
    return _translate_single(
        image, bboxes,
        translate_height, translate_width,
        np.shape, np.reshape, _np_convert, np.where, np.abs, np.concatenate,
        _np_logical_and, np.expand_dims, np.squeeze, _np_boolean_mask,
        _np_pad_images
    )


def translate(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        translate_heights: np.ndarray,
        translate_widths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    def fn(idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return _np_translate_single(
            images[idx], bboxes_ragged[idx],
            translate_heights[idx], translate_widths[idx]
        )

    return _np_map_idx_fn(fn, int(np.shape(images)[0]))


class Resize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return resize(images, bboxes_ragged, self.dest_size)


class RandomTransform:

    def __init__(
            self,
            np_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
            batch_size: int,
            probability: float,
            seed: int,
    ) -> None:
        self._np_fn = np_fn
        self.probability = probability
        self._rng = np.random.default_rng(seed=seed)
        self._batch_rand_fn: Callable[..., np.ndarray] = \
            lambda: self._rng.random(batch_size)

    def call(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:

        transformed_images, transformed_bboxes_ragged = self._np_fn(
            images, bboxes_ragged, *args, **kwargs
        )

        is_used = self._batch_rand_fn() < self.probability

        final_images = np.where(
            is_used, transformed_images, images
        )
        final_bboxes_ragged = np.where(
            is_used, transformed_bboxes_ragged, bboxes_ragged
        )

        return final_images, final_bboxes_ragged


class RandomFlipLeftRight(RandomTransform):

    def __init__(
            self,
            batch_size: int,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(flip_left_right, batch_size, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomFlipUpDown(RandomTransform):

    def __init__(
            self,
            batch_size: int,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(flip_up_down, batch_size, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomRotate90(RandomTransform):

    def __init__(
            self,
            batch_size: int,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(rotate_90, batch_size, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomRotate90AndResize(RandomTransform):

    def __init__(
            self,
            batch_size: int,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(rotate_90_and_resize, batch_size, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomRotate(RandomTransform):

    def __init__(
            self,
            batch_size: int,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(rotate, batch_size, probability, seed)
        assert angle_deg_range[0] < angle_deg_range[1]
        self.angle_deg_range = angle_deg_range

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        angles_deg = \
            self.angle_deg_range[1] - self.angle_deg_range[0] \
            * self._batch_rand_fn() + self.angle_deg_range[0]

        return super().call(images, bboxes_ragged, angles_deg)


class RandomShear(RandomTransform):

    def __init__(
            self,
            batch_size: int,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(shear, batch_size, probability, seed)
        assert -90.0 < angle_deg_range[0] < angle_deg_range[1] < 90.0
        self.angle_deg_range = angle_deg_range

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        angles_deg = \
            self.angle_deg_range[1] - self.angle_deg_range[0] \
            * self._batch_rand_fn() + self.angle_deg_range[0]

        return super().call(images, bboxes_ragged, angles_deg)


class RandomCropAndResize(RandomTransform):

    def __init__(
            self,
            batch_size: int,
            crop_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(crop_and_resize, batch_size, probability, seed)
        self.crop_height_fraction_range = crop_height_fraction_range
        self.crop_width_fraction_range = crop_width_fraction_range

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        offset_heights, offset_widths, cropped_heights, cropped_widths = \
            _np_get_random_crop_inputs(
                np.shape(images)[1], np.shape(images)[2],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                self._batch_rand_fn
            )

        return super().call(
            images, bboxes_ragged,
            offset_heights, offset_widths, cropped_heights, cropped_widths
        )


class RandomTranslate(RandomTransform):

    def __init__(
            self,
            batch_size: int,
            translate_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            translate_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(translate, batch_size, probability, seed)
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        height_fractions, width_fractions = _get_random_size_fractions(
            self.translate_height_fraction_range,
            self.translate_width_fraction_range,
            self._batch_rand_fn, _np_convert
        )

        translate_heights = np.shape(images)[1] * height_fractions
        translate_widths = np.shape(images)[2] * width_fractions

        return super().call(
            images, bboxes_ragged, translate_heights, translate_widths
        )
