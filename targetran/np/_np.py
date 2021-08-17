"""
API for Numpy usage.
"""

import functools
from typing import Any, Callable, Optional, Sequence, Tuple

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
    _affine_transform,
    _flip_left_right,
    _flip_up_down,
    _rotate,
    _shear,
    _translate,
    _get_crop_inputs,
    _get_random_size_fractions,
    _crop,
    _resize,
    _get_flip_left_right_mats,
    _get_flip_up_down_mats,
    _get_rotate_mats,
    _get_shear_mats,
    _get_translate_mats
)


def _np_affine_transform(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        image_dest_tran_mat: np.ndarray,
        bboxes_tran_mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _affine_transform(
        image, bboxes, labels,
        _np_convert, np.shape, np.reshape, np.expand_dims, np.squeeze,
        _np_pad_image, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.ones_like, np.stack, np.concatenate,
        image_dest_tran_mat, bboxes_tran_mat, np.matmul, np.clip,
        _np_gather_image, np.copy, np.max, np.min,
        _np_logical_and, _np_boolean_mask
    )


def flip_left_right(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _flip_left_right(
        image, bboxes, labels,
        _np_convert, np.shape, np.reshape, np.expand_dims, np.squeeze,
        _np_pad_image, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.ones_like, np.stack, np.concatenate, np.matmul, np.clip,
        _np_gather_image, np.copy, np.max, np.min,
        _np_logical_and, _np_boolean_mask
    )


def flip_up_down(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _flip_up_down(
        image, bboxes, labels,
        _np_convert, np.shape, np.reshape, np.expand_dims, np.squeeze,
        _np_pad_image, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.ones_like, np.stack, np.concatenate, np.matmul, np.clip,
        _np_gather_image, np.copy, np.max, np.min,
        _np_logical_and, _np_boolean_mask
    )


def rotate(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        angle_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _rotate(
        image, bboxes, labels, _np_convert(angle_deg),
        _np_convert, np.cos, np.sin, np.shape, np.reshape,
        np.expand_dims, np.squeeze, _np_pad_image, _np_range,
        _np_round_to_int, np.repeat, np.tile,
        np.ones_like, np.stack, np.concatenate, np.matmul,
        np.clip, _np_gather_image, np.copy, np.max, np.min,
        _np_logical_and, _np_boolean_mask
    )


def shear(
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
        angle_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _shear(
        image, bboxes, labels, _np_convert(angle_deg),
        _np_convert, np.tan, np.shape, np.reshape, np.expand_dims, np.squeeze,
        _np_pad_image, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.ones_like, np.stack, np.concatenate, np.matmul, np.clip,
        _np_gather_image, np.copy, np.max, np.min,
        _np_logical_and, _np_boolean_mask
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
        _np_convert(translate_height), _np_convert(translate_width),
        _np_convert, np.shape, np.reshape, np.expand_dims, np.squeeze,
        _np_pad_image, _np_range, _np_round_to_int, np.repeat, np.tile,
        np.ones_like, np.stack, np.concatenate, np.matmul, np.clip,
        _np_gather_image, np.copy, np.max, np.min,
        _np_logical_and, _np_boolean_mask
    )


def _np_get_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _get_crop_inputs(
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
        _np_convert(offset_height), _np_convert(offset_width),
        _np_convert(cropped_image_height), _np_convert(cropped_image_width),
        _np_convert, np.shape, np.reshape, np.concatenate,
        _np_logical_and, np.squeeze, np.clip, _np_boolean_mask
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

    def _rand_fn(self, shape: Tuple[int, ...] = ()) -> np.ndarray:
        return self._rng.random(shape)

    def get_mats(
            self,
            image: np.ndarray,
            rand_fn: Callable[..., np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Make sure inputs are in the needed format.
        image = _np_convert(image)
        bboxes = _np_convert(bboxes)
        labels = _np_convert(labels)

        if self._rand_fn() < self.probability:
            return self._np_fn(image, bboxes, labels, *args, **kwargs)
        return image, bboxes, labels


class CombineAffine(RandomTransform):

    def __init__(
            self,
            transforms: Sequence[RandomTransform],
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(_np_affine_transform, probability, seed)
        self._transforms = transforms
        self._identity_mat = np.expand_dims(np.array([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        ]), axis=0)

    def _combine_mats(
            self,
            image: np.ndarray,
            rand_fn: Callable[..., np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        image_dest_tran_mats, bboxes_tran_mats, probs = tuple(zip(
            *[(*t.get_mats(image, rand_fn), t.probability)
              for i, t in enumerate(self._transforms)]
        ))

        conditions = np.reshape(rand_fn() < probs, (len(probs), 1, 1))
        image_dest_tran_mats = np.where(
            conditions, image_dest_tran_mats, self._identity_mat
        )
        bboxes_tran_mats = np.where(
            conditions, bboxes_tran_mats, self._identity_mat
        )

        image_dest_tran_mat = functools.reduce(
            np.matmul, image_dest_tran_mats
        )
        # Note the reversed order for the bboxes tran matrices.
        bboxes_tran_mat = functools.reduce(
            np.matmul, bboxes_tran_mats[::-1]
        )
        return image_dest_tran_mat, bboxes_tran_mat

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_dest_tran_mat, bboxes_tran_mat = self._combine_mats(
            image, self._rand_fn
        )
        return super().__call__(
            image, bboxes, labels, image_dest_tran_mat, bboxes_tran_mat
        )


class RandomFlipLeftRight(RandomTransform):

    def __init__(
            self,
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(flip_left_right, probability, seed)

    def get_mats(
            self,
            image: np.ndarray,
            rand_fn: Callable[..., np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _get_flip_left_right_mats(_np_convert)

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

    def get_mats(
            self,
            image: np.ndarray,
            rand_fn: Callable[..., np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _get_flip_up_down_mats(_np_convert)

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

    def _get_angle_deg(self, rand_fn: Callable[..., np.ndarray]) -> np.ndarray:
        return self.angle_deg_range[1] - self.angle_deg_range[0] \
               * rand_fn() + self.angle_deg_range[0]

    def get_mats(
            self,
            image: np.ndarray,
            rand_fn: Callable[..., np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _get_rotate_mats(
            self._get_angle_deg(rand_fn), _np_convert, np.cos, np.sin
        )

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().__call__(
            image, bboxes, labels, self._get_angle_deg(self._rand_fn)
        )


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

    def _get_angle_deg(self, rand_fn: Callable[..., np.ndarray]) -> np.ndarray:
        return self.angle_deg_range[1] - self.angle_deg_range[0] \
               * rand_fn() + self.angle_deg_range[0]

    def get_mats(
            self,
            image: np.ndarray,
            rand_fn: Callable[..., np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _get_shear_mats(self._get_angle_deg(rand_fn), _np_convert, np.tan)

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().__call__(
            image, bboxes, labels, self._get_angle_deg(self._rand_fn)
        )


class RandomTranslate(RandomTransform):

    def __init__(
            self,
            translate_height_fraction_range: Tuple[float, float] = (-0.2, 0.2),
            translate_width_fraction_range: Tuple[float, float] = (-0.2, 0.2),
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(translate, probability, seed)
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range

    def _get_translate_height_and_width(
            self,
            image: np.ndarray,
            rand_fn: Callable[..., np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        height_fraction, width_fraction = _get_random_size_fractions(
            self.translate_height_fraction_range,
            self.translate_width_fraction_range,
            rand_fn, _np_convert
        )
        translate_height = _np_round_to_int(
            np.shape(image)[0] * height_fraction
        )
        translate_width = _np_round_to_int(
            np.shape(image)[1] * width_fraction
        )
        return translate_height, translate_width

    def get_mats(
            self,
            image: np.ndarray,
            rand_fn: Callable[..., np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        translate_height, translate_width = \
            self._get_translate_height_and_width(image, rand_fn)
        return _get_translate_mats(
            translate_height, translate_width, _np_convert
        )

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        translate_height, translate_width = \
            self._get_translate_height_and_width(image, self._rand_fn)
        return super().__call__(
            image, bboxes, labels, translate_height, translate_width
        )


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
            _np_get_crop_inputs(
                np.shape(image)[0], np.shape(image)[1],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                self._rand_fn
            )

        return super().__call__(
            image, bboxes, labels,
            offset_height, offset_width, cropped_height, cropped_width
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
