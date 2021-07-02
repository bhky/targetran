"""
API for TensorFlow usage.
"""

from typing import Any, Callable, List, Tuple

import tensorflow as tf

from ._transform import (
    _tf_resize,
    _tf_flip_left_right,
    _tf_flip_up_down,
    _tf_rotate_90,
    _tf_rotate_90_and_pad_and_resize,
    _tf_crop_and_resize
)


class TFResize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return _tf_resize(images, bboxes_list, self.dest_size)


class TFRandomTransform:

    def __init__(
            self,
            tf_fn: Callable[..., Tuple[tf.Tensor, List[tf.Tensor]]],
            probability: float,
            seed: int,
    ) -> None:
        self._tf_fn = tf_fn
        self.probability = probability
        self.seed = seed

    def call(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor],
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:

        transformed_images, transformed_bboxes_list = self._tf_fn(
            images, bboxes_list, *args, **kwargs
        )

        rand = tf.random.uniform(shape=tf.shape(images)[0], seed=self.seed)
        is_used = rand < self.probability

        final_images = tf.where(is_used, transformed_images, images)
        final_bboxes_list = [
            transformed_bboxes_list[i] if is_used[i] else bboxes_list[i]
            for i in range(len(bboxes_list))
        ]

        return final_images, final_bboxes_list


class TFRandomFlipLeftRight(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_flip_left_right, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(images, bboxes_list)


class TFRandomFlipUpDown(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_flip_up_down, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(images, bboxes_list)


class TFRandomRotate90(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_rotate_90, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(images, bboxes_list)


class TFRandomRotate90AndResize(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_rotate_90_and_pad_and_resize, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        return super().call(images, bboxes_list)


class TFRandomCropAndResize(TFRandomTransform):

    def __init__(
            self,
            max_x_offset_fraction: float = 0.2,
            max_y_offset_fraction: float = 0.2,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(_tf_crop_and_resize, probability, seed)
        self.max_x_offset_fraction = max_x_offset_fraction
        self.max_y_offset_fraction = max_y_offset_fraction

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_list: List[tf.Tensor]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        batch_size = tf.shape(images)[0]
        return super().call(
            images,
            bboxes_list,
            tf.random.uniform(
                shape=[batch_size], maxval=self.max_x_offset_fraction
            ),
            tf.random.uniform(
                shape=[batch_size], maxval=self.max_y_offset_fraction
            )
        )
