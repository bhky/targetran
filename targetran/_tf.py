"""
API for TensorFlow usage.
"""

from typing import Any, Callable, List, Tuple

import tensorflow as tf

from ._transform import (
    _tf_flip_left_right,
    _tf_flip_up_down,
    _tf_rotate_90,
    _tf_crop_and_resize
)


class TFRandomTransform:

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


class TFRandomFlipLeftRight(TFRandomTransform):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_flip_left_right, flip_probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(images, bboxes_list)


class TFRandomFlipUpDown(TFRandomTransform):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_flip_up_down, flip_probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(images, bboxes_list)


class TFRandomRotate90(TFRandomTransform):

    def __init__(self, flip_probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_rotate_90, flip_probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(images, bboxes_list)


class TFRandomCropAndResize(TFRandomTransform):

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
