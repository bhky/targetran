"""
API for Numpy usage.
"""

from typing import Any, Callable, List, Tuple

import numpy as np

from ._transform import (
    _np_flip_left_right,
    _np_flip_up_down,
    _np_rotate_90,
    _np_crop_and_resize
)


class RandomTransform:

    def __init__(
            self,
            np_fn: Callable[..., Tuple[np.ndarray, List[np.ndarray]]],
            flip_probability: float,
            seed: int,
    ) -> None:
        self._np_fn = np_fn
        self.flip_probability = flip_probability
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def call(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray],
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, List[np.ndarray]]:

        rand = self.rng.random(size=np.shape(images)[:1])
        output: Tuple[np.ndarray, List[np.ndarray]] = np.where(
            np.less(rand, self.flip_probability),
            self._np_fn(images, bboxes_list, *args, **kwargs),
            (images, bboxes_list)
        )
        return output


class RandomFlipLeftRight(RandomTransform):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_flip_left_right, flip_probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        return super().call(images, bboxes_list)


class RandomFlipUpDown(RandomTransform):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_flip_up_down, flip_probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        return super().call(images, bboxes_list)


class RandomRotate90(RandomTransform):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_rotate_90, flip_probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        return super().call(images, bboxes_list)


class RandomCropAndResize(RandomTransform):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_crop_and_resize, flip_probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray],
            x_offset_fractions: np.ndarray,
            y_offset_fractions: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        return super().call(
            images,
            bboxes_list,
            x_offset_fractions,
            y_offset_fractions
        )
