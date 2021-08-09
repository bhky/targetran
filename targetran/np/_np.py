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
        _np_convert, np.shape, np.reshape, np.concatenate
    )


def flip_up_down(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _flip_up_down(
        image, bboxes, labels,
        _np_convert, np.shape, np.reshape, np.concatenate
    )


def resize(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        dest_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _resize(
        image, bboxes, labels, dest_size,
        _np_convert, np.shape, np.reshape, _np_resize_image, np.concatenate
    )


def rotate_90(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _rotate_90(
        image, bboxes, labels,
        _np_convert, np.shape, np.reshape, np.transpose, np.concatenate
    )


def rotate_90_and_pad(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _rotate_90_and_pad(
        image, bboxes, labels,
        _np_convert, np.shape, np.reshape, np.transpose, np.concatenate,
        np.where, np.ceil, np.floor, _np_pad_image
    )


def rotate(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        angle_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _rotate(
        image, bboxes, labels, angle_deg,
        _np_convert, np.shape, np.reshape, np.expand_dims, np.squeeze,
        _np_pad_image, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.stack, np.concatenate, np.cos, np.sin, np.matmul, np.clip,
        _np_gather_image, np.copy, np.max, np.min,
        _np_logical_and, _np_boolean_mask
    )


def shear(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        angle_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _shear(
        image, bboxes, labels, angle_deg,
        _np_convert, np.shape, np.reshape, np.expand_dims, np.squeeze,
        _np_pad_image, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.stack, np.concatenate, np.tan, np.matmul, np.clip,
        _np_gather_image, np.copy, np.max, np.min,
        _np_logical_and, _np_boolean_mask
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


def crop(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        offset_height: int,
        offset_width: int,
        cropped_image_height: int,
        cropped_image_width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _crop(
        image, bboxes, labels,
        offset_height, offset_width,
        cropped_image_height, cropped_image_width,
        _np_convert, np.shape, np.reshape, np.concatenate,
        _np_logical_and, np.squeeze, np.clip, _np_boolean_mask
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
        _np_convert, np.shape, np.reshape, np.where, np.abs, np.concatenate,
        _np_logical_and, np.squeeze, np.clip, _np_boolean_mask, _np_pad_image
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
            seed: Optional[int],
    ) -> None:
        self._np_fn = np_fn
        self.probability = probability
        self._rng = np.random.default_rng(seed=seed)
        self._rand_fn: Callable[..., np.ndarray] = lambda: self._rng.random()

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        image = _np_convert(image)
        bboxes = _np_convert(bboxes)
        labels = _np_convert(labels)

        if self._rand_fn() < self.probability:
            return self._np_fn(image, bboxes, labels, *args, **kwargs)
        return image, bboxes, labels


class RandomFlipLeftRight(RandomTransform):

    def __init__(
            self,
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(flip_left_right, probability, seed)

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().__call__(image, bboxes, labels)


class RandomFlipUpDown(RandomTransform):

    def __init__(
            self,
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(flip_up_down, probability, seed)

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().__call__(image, bboxes, labels)


class RandomRotate90(RandomTransform):

    def __init__(
            self,
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(rotate_90, probability, seed)

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().__call__(image, bboxes, labels)


class RandomRotate90AndPad(RandomTransform):

    def __init__(
            self,
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(rotate_90_and_pad, probability, seed)

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().__call__(image, bboxes, labels)


class RandomRotate(RandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(rotate, probability, seed)
        assert angle_deg_range[0] < angle_deg_range[1]
        self.angle_deg_range = angle_deg_range

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        angle_deg = \
            self.angle_deg_range[1] - self.angle_deg_range[0] \
            * self._rand_fn() + self.angle_deg_range[0]

        return super().__call__(image, bboxes, labels, angle_deg)


class RandomShear(RandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(shear, probability, seed)
        assert -90.0 < angle_deg_range[0] < angle_deg_range[1] < 90.0
        self.angle_deg_range = angle_deg_range

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        angle_deg = \
            self.angle_deg_range[1] - self.angle_deg_range[0] \
            * self._rand_fn() + self.angle_deg_range[0]

        return super().__call__(image, bboxes, labels, angle_deg)


class RandomCrop(RandomTransform):

    def __init__(
            self,
            crop_height_fraction_range: Tuple[float, float] = (0.7, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.7, 0.9),
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(crop, probability, seed)
        self.crop_height_fraction_range = crop_height_fraction_range
        self.crop_width_fraction_range = crop_width_fraction_range

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        offset_height, offset_width, cropped_height, cropped_width = \
            _np_get_random_crop_inputs(
                np.shape(image)[0], np.shape(image)[1],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                self._rand_fn
            )

        return super().__call__(
            image, bboxes, labels,
            offset_height, offset_width, cropped_height, cropped_width
        )


class RandomTranslate(RandomTransform):

    def __init__(
            self,
            translate_height_fraction_range: Tuple[float, float] = (-0.1, 0.1),
            translate_width_fraction_range: Tuple[float, float] = (-0.1, 0.1),
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(translate, probability, seed)
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        height_fraction, width_fraction = _get_random_size_fractions(
            self.translate_height_fraction_range,
            self.translate_width_fraction_range,
            self._rand_fn, _np_convert
        )

        translate_height = _np_round_to_int(
            np.shape(image)[0] * height_fraction
        )
        translate_width = _np_round_to_int(
            np.shape(image)[1] * width_fraction
        )

        return super().__call__(
            image, bboxes, labels, translate_height, translate_width
        )
