"""
API for Numpy usage.
"""

from typing import Any, Callable, Tuple

import numpy as np  # type: ignore

from ._functional import _np_convert
from ._transform import (
    _get_random_size_fractions,
    _np_resize,
    _np_flip_left_right,
    _np_flip_up_down,
    _np_rotate_90,
    _np_rotate_90_and_pad_and_resize,
    _np_get_random_crop_inputs,
    _np_crop_and_resize,
    _np_translate
)


class Resize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _np_resize(images, bboxes_ragged, self.dest_size)


class RandomTransform:

    def __init__(
            self,
            np_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
            probability: float,
            seed: int,
    ) -> None:
        self._np_fn = np_fn
        self.probability = probability
        self.rng = np.random.default_rng(seed=seed)

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

        rand = self.rng.random(size=np.shape(images)[0])
        is_used = rand < self.probability

        final_images = np.where(is_used, transformed_images, images)
        final_bboxes_ragged_list = [
            transformed_bboxes_ragged[i] if is_used[i] else bboxes_ragged[i]
            for i in range(len(bboxes_ragged))
        ]

        return final_images, np.array(final_bboxes_ragged_list, dtype=object)


class RandomFlipLeftRight(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_flip_left_right, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomFlipUpDown(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_flip_up_down, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomRotate90(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_rotate_90, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomRotate90AndResize(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_rotate_90_and_pad_and_resize, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomCropAndResize(RandomTransform):

    def __init__(
            self,
            crop_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(_np_crop_and_resize, probability, seed)
        self.crop_height_fraction_range = crop_height_fraction_range
        self.crop_width_fraction_range = crop_width_fraction_range

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        images_shape = np.shape(images)

        def rand_fn() -> np.ndarray:
            return self.rng.random(images_shape[0])

        offset_heights, offset_widths, cropped_heights, cropped_widths = \
            _np_get_random_crop_inputs(
                images_shape[1], images_shape[2],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                rand_fn
            )

        return super().call(
            images, bboxes_ragged,
            offset_heights, offset_widths, cropped_heights, cropped_widths
        )


class RandomTranslate(RandomTransform):

    def __init__(
            self,
            translate_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            translate_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(_np_translate, probability, seed)
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        images_shape = np.shape(images)

        def rand_fn() -> np.ndarray:
            return self.rng.random(images_shape[0])

        height_fractions, width_fractions = _get_random_size_fractions(
            self.translate_height_fraction_range,
            self.translate_width_fraction_range,
            rand_fn, _np_convert
        )

        translate_heights = images_shape[1] * height_fractions
        translate_widths = images_shape[2] * width_fractions

        return super().call(
            images, bboxes_ragged, translate_heights, translate_widths
        )
