"""
API for NumPy usage.
"""
import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from targetran._check import (
    _check_shear_input,
    _check_translate_input,
    _check_crop_input,
    _check_input_range,
)
from targetran._np_functional import (
    _np_convert,
    _np_range,
    _np_cast_to_int,
    _np_round_to_int,
    _np_resize_image,
    _np_boolean_mask,
    _np_logical_and,
    _np_pad_image,
    _np_gather_image,
)
from targetran._transform import (
    _AffineDependency,
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
    _get_translate_mats,
)
from targetran._typing import NDFloatArray, NDIntArray
from targetran.utils import Interpolation


def _np_get_affine_dependency() -> _AffineDependency:
    return _AffineDependency(
        _np_convert, np.shape, np.reshape, np.expand_dims, np.squeeze,
        _np_pad_image, _np_range, _np_cast_to_int, _np_round_to_int, np.repeat,
        np.tile,  # type: ignore
        np.ones_like, np.stack, np.concatenate, np.matmul,
        np.clip, np.floor, np.ceil, _np_gather_image,
        np.copy, np.max, np.min,
        _np_logical_and, _np_boolean_mask
    )


def _np_affine_transform(
        image: NDFloatArray,
        bboxes: NDFloatArray,
        labels: NDFloatArray,
        image_dest_tran_mat: NDFloatArray,
        bboxes_tran_mat: NDFloatArray,
        interpolation: Interpolation
) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
    return _affine_transform(
        image, bboxes, labels, image_dest_tran_mat, bboxes_tran_mat,
        interpolation, _np_get_affine_dependency()
    )


def flip_left_right(
        image: NDFloatArray,
        bboxes: NDFloatArray,
        labels: NDFloatArray
) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
    return _flip_left_right(
        image, bboxes, labels,
        Interpolation.NEAREST, _np_get_affine_dependency()
    )


def flip_up_down(
        image: NDFloatArray,
        bboxes: NDFloatArray,
        labels: NDFloatArray
) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
    return _flip_up_down(
        image, bboxes, labels,
        Interpolation.NEAREST, _np_get_affine_dependency()
    )


def rotate(
        image: NDFloatArray,
        bboxes: NDFloatArray,
        labels: NDFloatArray,
        angle_deg: float,
        interpolation: Interpolation = Interpolation.BILINEAR
) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
    return _rotate(
        image, bboxes, labels, _np_convert(angle_deg), np.cos, np.sin,
        interpolation, _np_get_affine_dependency()
    )


def shear(
        image: NDFloatArray,
        bboxes: NDFloatArray,
        labels: NDFloatArray,
        angle_deg: float,
        interpolation: Interpolation = Interpolation.BILINEAR,
        _check_input: bool = True
) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
    if _check_input:
        _check_shear_input(angle_deg)
    return _shear(
        image, bboxes, labels, _np_convert(angle_deg), np.tan,
        interpolation, _np_get_affine_dependency()
    )


def translate(
        image: NDFloatArray,
        bboxes: NDFloatArray,
        labels: NDFloatArray,
        translate_height: int,
        translate_width: int,
        interpolation: Interpolation = Interpolation.BILINEAR,
        _check_input: bool = True
) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
    if _check_input:
        _check_translate_input(image.shape, translate_height, translate_width)
    return _translate(
        image, bboxes, labels,
        _np_convert(translate_height), _np_convert(translate_width),
        interpolation, _np_get_affine_dependency()
    )


def _np_get_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., NDFloatArray]
) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray, NDFloatArray]:
    return _get_crop_inputs(
        image_height, image_width, height_fraction_range, width_fraction_range,
        rand_fn, _np_convert, _np_round_to_int
    )


def crop(
        image: NDFloatArray,
        bboxes: NDFloatArray,
        labels: NDFloatArray,
        offset_height: int,
        offset_width: int,
        crop_height: int,
        crop_width: int,
        _check_input: bool = True
) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
    if _check_input:
        _check_crop_input(image.shape, offset_height, offset_width)
    return _crop(
        image, bboxes, labels,
        _np_convert(offset_height), _np_convert(offset_width),
        _np_convert(crop_height), _np_convert(crop_width),
        _np_convert, np.shape, np.reshape, np.concatenate,
        _np_logical_and, np.squeeze, np.clip, _np_boolean_mask
    )


def resize(
        image: NDFloatArray,
        bboxes: NDFloatArray,
        labels: NDFloatArray,
        dest_size: Tuple[int, int]
) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
    return _resize(
        image, bboxes, labels, dest_size,
        _np_convert, np.shape, np.reshape, _np_resize_image, np.concatenate
    )


class RandomTransform:

    def __init__(
            self,
            np_fn: Callable[..., Tuple[NDFloatArray, NDFloatArray, NDFloatArray]],
            probability: float,
            seed: Optional[int],
            name: str,
            is_affine: bool
    ) -> None:
        self._np_fn = np_fn
        self.probability = probability
        self._rng = np.random.default_rng(seed=seed)
        self.name = name
        self.is_affine = is_affine

    def _rand_fn(self, shape: Sequence[int] = ()) -> NDFloatArray:
        return self._rng.random(shape)

    def _get_mats(
            self,
            image: NDFloatArray,
            rand_fn: Callable[..., NDFloatArray]
    ) -> Tuple[NDFloatArray, NDFloatArray]:
        pass

    def __call__(
            self,
            image: NDFloatArray,
            bboxes: NDFloatArray,
            labels: NDFloatArray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
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
            num_selected_transforms: Optional[int] = None,
            selected_probabilities: Optional[List[float]] = None,
            interpolation: Interpolation = Interpolation.BILINEAR,
            probability: float = 1.0,
            seed: Optional[int] = None
    ) -> None:
        not_affine_trans = list(filter(lambda t: not t.is_affine, transforms))
        if not_affine_trans:
            raise AssertionError(
                f"Non-affine transforms cannot be included in CombineAffine: "
                f"{[t.name for t in not_affine_trans]}"
            )
        if num_selected_transforms and selected_probabilities:
            if len(selected_probabilities) != len(transforms):
                raise ValueError(
                    "Number of items in selected_probabilities should be "
                    "the same as the number of items in transforms."
                )
        super().__init__(
            _np_affine_transform, probability, seed, "CombineAffine", True
        )
        self._transforms = transforms
        self._num_selected_transforms = num_selected_transforms
        self._selected_probabilities = selected_probabilities
        self._interpolation = interpolation
        self._identity_mat = np.expand_dims(np.array([  # type: ignore
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        ]), axis=0)

    def _get_mats(
            self,
            image: NDFloatArray,
            rand_fn: Callable[..., NDFloatArray]
    ) -> Tuple[NDFloatArray, NDFloatArray]:
        image_dest_tran_mats, bboxes_tran_mats, probs = tuple(zip(
            *[(*t._get_mats(image, rand_fn), t.probability)
              for i, t in enumerate(self._transforms)]
        ))

        if self._num_selected_transforms:
            indices = self._rng.choice(
                len(self._transforms),
                self._num_selected_transforms,
                replace=False, p=self._selected_probabilities
            ).tolist()
            image_dest_tran_mats = np.take(image_dest_tran_mats, indices, 0)
            bboxes_tran_mats = np.take(bboxes_tran_mats, indices, 0)
        else:
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
            image: NDFloatArray,
            bboxes: NDFloatArray,
            labels: NDFloatArray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
        image_dest_tran_mat, bboxes_tran_mat = self._get_mats(
            image, self._rand_fn
        )
        return super().__call__(
            image, bboxes, labels, image_dest_tran_mat, bboxes_tran_mat,
            self._interpolation
        )


class RandomFlipLeftRight(RandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(
            flip_left_right, probability, seed, "RandomFlipLeftRight", True
        )

    def _get_mats(
            self,
            image: NDFloatArray,
            rand_fn: Callable[..., NDFloatArray]
    ) -> Tuple[NDFloatArray, NDFloatArray]:
        return _get_flip_left_right_mats(_np_convert)

    def __call__(
            self,
            image: NDFloatArray,
            bboxes: NDFloatArray,
            labels: NDFloatArray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
        return super().__call__(image, bboxes, labels)


class RandomFlipUpDown(RandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(
            flip_up_down, probability, seed, "RandomFlipUpDown", True
        )

    def _get_mats(
            self,
            image: NDFloatArray,
            rand_fn: Callable[..., NDFloatArray]
    ) -> Tuple[NDFloatArray, NDFloatArray]:
        return _get_flip_up_down_mats(_np_convert)

    def __call__(
            self,
            image: NDFloatArray,
            bboxes: NDFloatArray,
            labels: NDFloatArray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
        return super().__call__(image, bboxes, labels)


class RandomRotate(RandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            interpolation: Interpolation = Interpolation.BILINEAR,
            probability: float = 0.9,
            seed: Optional[int] = None
    ) -> None:
        _check_input_range(angle_deg_range, None, "angle_deg_range")
        super().__init__(rotate, probability, seed, "RandomRotate", True)
        self.angle_deg_range: NDFloatArray = np.array(angle_deg_range)
        self.interpolation = interpolation

    def _get_angle_deg(
            self,
            rand_fn: Callable[..., NDFloatArray]
    ) -> NDFloatArray:
        return (  # type: ignore
            self.angle_deg_range[1] - self.angle_deg_range[0]
            * rand_fn() + self.angle_deg_range[0]
        )

    def _get_mats(
            self,
            image: NDFloatArray,
            rand_fn: Callable[..., NDFloatArray]
    ) -> Tuple[NDFloatArray, NDFloatArray]:
        return _get_rotate_mats(
            self._get_angle_deg(rand_fn), _np_convert, np.cos, np.sin
        )

    def __call__(
            self,
            image: NDFloatArray,
            bboxes: NDFloatArray,
            labels: NDFloatArray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
        return super().__call__(
            image, bboxes, labels, self._get_angle_deg(self._rand_fn),
            self.interpolation
        )


class RandomShear(RandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-10.0, 10.0),
            interpolation: Interpolation = Interpolation.BILINEAR,
            probability: float = 0.9,
            seed: Optional[int] = None
    ) -> None:
        _check_input_range(angle_deg_range, (-90.0, 90.0), "angle_deg_range")
        super().__init__(shear, probability, seed, "RandomShear", True)
        self.angle_deg_range: NDFloatArray = np.array(angle_deg_range)
        self.interpolation = interpolation

    def _get_angle_deg(
            self,
            rand_fn: Callable[..., NDFloatArray]
    ) -> NDFloatArray:
        return (  # type: ignore
            self.angle_deg_range[1] - self.angle_deg_range[0]
            * rand_fn() + self.angle_deg_range[0]
        )

    def _get_mats(
            self,
            image: NDFloatArray,
            rand_fn: Callable[..., NDFloatArray]
    ) -> Tuple[NDFloatArray, NDFloatArray]:
        return _get_shear_mats(self._get_angle_deg(rand_fn), _np_convert, np.tan)

    def __call__(
            self,
            image: NDFloatArray,
            bboxes: NDFloatArray,
            labels: NDFloatArray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
        return super().__call__(
            image, bboxes, labels, self._get_angle_deg(self._rand_fn),
            self.interpolation, False
        )


class RandomTranslate(RandomTransform):

    def __init__(
            self,
            translate_height_fraction_range: Tuple[float, float] = (-0.1, 0.1),
            translate_width_fraction_range: Tuple[float, float] = (-0.1, 0.1),
            interpolation: Interpolation = Interpolation.BILINEAR,
            probability: float = 0.9,
            seed: Optional[int] = None
    ) -> None:
        _check_input_range(
            translate_height_fraction_range, (-1.0, 1.0),
            "translate_height_fraction_range"
        )
        _check_input_range(
            translate_width_fraction_range, (-1.0, 1.0),
            "translate_width_fraction_range"
        )
        super().__init__(translate, probability, seed, "RandomTranslate", True)
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range
        self.interpolation = interpolation

    def _get_translate_height_and_width(
            self,
            image: NDFloatArray,
            rand_fn: Callable[..., NDFloatArray]
    ) -> Tuple[NDIntArray, NDIntArray]:
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

    def _get_mats(
            self,
            image: NDFloatArray,
            rand_fn: Callable[..., NDFloatArray]
    ) -> Tuple[NDFloatArray, NDFloatArray]:
        translate_height, translate_width = \
            self._get_translate_height_and_width(image, rand_fn)
        return _get_translate_mats(
            translate_height, translate_width, _np_convert
        )

    def __call__(
            self,
            image: NDFloatArray,
            bboxes: NDFloatArray,
            labels: NDFloatArray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
        translate_height, translate_width = \
            self._get_translate_height_and_width(image, self._rand_fn)
        return super().__call__(
            image, bboxes, labels, translate_height, translate_width,
            self.interpolation, False
        )


class RandomCrop(RandomTransform):

    def __init__(
            self,
            crop_height_fraction_range: Tuple[float, float] = (0.8, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.8, 0.9),
            probability: float = 0.9,
            seed: Optional[int] = None
    ) -> None:
        _check_input_range(
            crop_height_fraction_range, (0.0, 1.0), "crop_height_fraction_range"
        )
        _check_input_range(
            crop_width_fraction_range, (0.0, 1.0), "crop_width_fraction_range"
        )
        super().__init__(crop, probability, seed, "RandomCrop", False)
        self.crop_height_fraction_range = crop_height_fraction_range
        self.crop_width_fraction_range = crop_width_fraction_range

    def __call__(
            self,
            image: NDFloatArray,
            bboxes: NDFloatArray,
            labels: NDFloatArray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
        offset_height, offset_width, crop_height, crop_width = \
            _np_get_crop_inputs(
                np.shape(image)[0], np.shape(image)[1],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                self._rand_fn
            )

        return super().__call__(
            image, bboxes, labels,
            offset_height, offset_width, crop_height, crop_width, False
        )


class Resize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size
        self.name = "Resize"
        self.is_affine = False

    def __call__(
            self,
            image: NDFloatArray,
            bboxes: NDFloatArray,
            labels: NDFloatArray
    ) -> Tuple[NDFloatArray, NDFloatArray, NDFloatArray]:
        return resize(image, bboxes, labels, self.dest_size)
