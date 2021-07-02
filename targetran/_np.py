"""
API for Numpy usage.
"""

from typing import Any, Callable, List, Tuple

import numpy as np

from ._transform import (
    _np_resize,
    _np_flip_left_right,
    _np_flip_up_down,
    _np_rotate_90,
    _np_rotate_90_and_pad_and_resize,
    _np_crop_and_resize
)


class Resize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size

    def __call__(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        return _np_resize(images, bboxes_list, self.dest_size)


class RandomTransform:

    def __init__(
            self,
            np_fn: Callable[..., Tuple[np.ndarray, List[np.ndarray]]],
            probability: float,
            seed: int,
    ) -> None:
        self._np_fn = np_fn
        self.probability = probability
        self.rng = np.random.default_rng(seed=seed)

    def call(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray],
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, List[np.ndarray]]:

        transformed_images, transformed_bboxes_list = self._np_fn(
            images, bboxes_list, *args, **kwargs
        )

        rand = self.rng.random(size=np.shape(images)[0])
        is_used = rand < self.probability

        final_images = np.where(is_used, transformed_images, images)
        final_bboxes_list = [
            transformed_bboxes_list[i] if is_used[i] else bboxes_list[i]
            for i in range(len(bboxes_list))
        ]

        return final_images, final_bboxes_list


class RandomFlipLeftRight(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_flip_left_right, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        return super().call(images, bboxes_list)


class RandomFlipUpDown(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_flip_up_down, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        return super().call(images, bboxes_list)


class RandomRotate90(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_rotate_90, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        return super().call(images, bboxes_list)


class RandomRotate90AndResize(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_rotate_90_and_pad_and_resize, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        return super().call(images, bboxes_list)


class RandomCropAndResize(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_np_crop_and_resize, probability, seed)

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
