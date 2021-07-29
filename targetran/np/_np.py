"""
API for Numpy usage.
"""

from typing import Any, Callable, Optional, Tuple

import numpy as np  # type: ignore

from targetran._functional import (
    _np_convert,
    _np_range,
    _np_round_to_int,
    _np_resize_image,
    _np_boolean_mask,
    _np_logical_and,
    _np_pad_image,
    _np_gather_image
)

from targetran._transform import (
    _flip_left_right,
    _flip_up_down,
    _rotate_90,
    _rotate_90_and_pad,
    _rotate,
    _shear,
    _crop,
    _resize,
    _translate,
    _get_random_crop_inputs,
    _get_random_size_fractions
)


def flip_left_right(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _flip_left_right(
        image, bboxes, labels,
        np.shape, _np_convert, np.concatenate
    )


def flip_up_down(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _flip_up_down(
        image, bboxes, labels,
        np.shape, _np_convert, np.concatenate
    )


def resize(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        dest_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _resize(
        image, bboxes, labels,
        dest_size, np.shape, _np_resize_image, _np_convert, np.concatenate
    )


def rotate_90(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _rotate_90(
        image, bboxes, labels,
        np.shape, _np_convert, np.transpose, np.concatenate
    )


def rotate_90_and_resize(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        dest_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Could be rotate_90_and_pad_and_resize, but thought it is too clumsy.

    If dest_size is None, use original image size.
    """
    if dest_size is None:
        height, width = int(np.shape(image)[0]), int(np.shape(image)[1])
    else:
        height, width = int(dest_size[0]), int(dest_size[1])

    image, bboxes, labels = _rotate_90_and_pad(
        image, bboxes, labels,
        np.shape, _np_convert, np.transpose, np.concatenate,
        np.where, np.ceil, np.floor, _np_pad_image
    )
    return resize(image, bboxes, labels, (height, width))


def rotate(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        angle_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _rotate(
        image, bboxes, labels, angle_deg,
        np.shape, _np_convert, np.expand_dims, np.squeeze,
        _np_pad_image, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.stack, np.concatenate, np.cos, np.sin, np.matmul, np.clip,
        _np_gather_image, np.reshape, np.copy,
        np.max, np.min, _np_logical_and, _np_boolean_mask
    )


def shear(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        angle_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _shear(
        image, bboxes, labels, angle_deg,
        np.shape, _np_convert, np.expand_dims, np.squeeze,
        _np_pad_image, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.stack, np.concatenate, np.tan, np.matmul, np.clip,
        _np_gather_image, np.reshape, np.copy,
        np.max, np.min, _np_logical_and, _np_boolean_mask
    )


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


def crop_and_resize(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        offset_height: int,
        offset_width: int,
        cropped_image_height: int,
        cropped_image_width: int,
        dest_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    If dest_size is None, use original image size.
    """
    dest_size = dest_size if dest_size is not None else np.shape(image)[0:2]
    cropped_image, cropped_bboxes, cropped_labels = _crop(
        image, bboxes, labels,
        offset_height, offset_width,
        cropped_image_height, cropped_image_width,
        np.shape, np.reshape, _np_convert, np.concatenate,
        _np_logical_and, np.squeeze, _np_boolean_mask
    )
    return resize(
        cropped_image, cropped_bboxes, cropped_labels, dest_size
    )


def translate(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        translate_height: int,
        translate_width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _translate(
        image, bboxes, labels,
        translate_height, translate_width,
        np.shape, np.reshape, _np_convert, np.where, np.abs, np.concatenate,
        _np_logical_and, np.squeeze, _np_boolean_mask, _np_pad_image
    )


class Resize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return resize(image, bboxes, labels, self.dest_size)


class RandomTransform:

    def __init__(
            self,
            np_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]],
            probability: float,
            seed: int,
    ) -> None:
        self._np_fn = np_fn
        self.probability = probability
        self._rng = np.random.default_rng(seed=seed)
        self._rand_fn: Callable[..., np.ndarray] = lambda: self._rng.random()

    def call(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self._rand_fn() < self.probability:
            return self._np_fn(image, bboxes, labels, *args, **kwargs)
        return image, bboxes, labels


class RandomFlipLeftRight(RandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(flip_left_right, probability, seed)

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().call(image, bboxes, labels)


class RandomFlipUpDown(RandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(flip_up_down, probability, seed)

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().call(image, bboxes, labels)


class RandomRotate90(RandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(rotate_90, probability, seed)

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().call(image, bboxes, labels)


class RandomRotate90AndResize(RandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(rotate_90_and_resize, probability, seed)

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().call(image, bboxes, labels)


class RandomRotate(RandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(rotate, probability, seed)
        assert angle_deg_range[0] < angle_deg_range[1]
        self.angle_deg_range = angle_deg_range

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        angle_deg = \
            self.angle_deg_range[1] - self.angle_deg_range[0] \
            * self._rand_fn() + self.angle_deg_range[0]

        return super().call(image, bboxes, labels, angle_deg)


class RandomShear(RandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(shear, probability, seed)
        assert -90.0 < angle_deg_range[0] < angle_deg_range[1] < 90.0
        self.angle_deg_range = angle_deg_range

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        angle_deg = \
            self.angle_deg_range[1] - self.angle_deg_range[0] \
            * self._rand_fn() + self.angle_deg_range[0]

        return super().call(image, bboxes, labels, angle_deg)


class RandomCropAndResize(RandomTransform):

    def __init__(
            self,
            crop_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(crop_and_resize, probability, seed)
        self.crop_height_fraction_range = crop_height_fraction_range
        self.crop_width_fraction_range = crop_width_fraction_range

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        offset_height, offset_width, cropped_height, cropped_width = \
            _np_get_random_crop_inputs(
                np.shape(image)[0], np.shape(image)[1],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                self._rand_fn
            )

        return super().call(
            image, bboxes, labels,
            offset_height, offset_width, cropped_height, cropped_width
        )


class RandomTranslate(RandomTransform):

    def __init__(
            self,
            translate_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            translate_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(translate, probability, seed)
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        height_fraction, width_fraction = _get_random_size_fractions(
            self.translate_height_fraction_range,
            self.translate_width_fraction_range,
            self._rand_fn, _np_convert
        )

        translate_height = np.shape(image)[0] * height_fraction
        translate_width = np.shape(image)[1] * width_fraction

        return super().call(
            image, bboxes, labels, translate_height, translate_width
        )