"""
API for TensorFlow usage.
"""

from typing import Any, Callable, Tuple

import tensorflow as tf

from ._transform import (
    _tf_resize,
    _tf_flip_left_right,
    _tf_flip_up_down,
    _tf_rotate_90,
    _tf_rotate_90_and_pad_and_resize,
    _tf_fractions_to_heights_and_widths,
    _tf_crop_and_resize
)


class TFResize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return _tf_resize(images, bboxes_ragged, self.dest_size)


class TFRandomTransform:

    def __init__(
            self,
            tf_fn: Callable[..., Tuple[tf.Tensor, tf.Tensor]],
            probability: float,
            seed: int,
    ) -> None:
        self._tf_fn = tf_fn
        self.probability = probability
        self.seed = seed

    def call(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        transformed_images, transformed_bboxes_ragged = self._tf_fn(
            images, bboxes_ragged, *args, **kwargs
        )

        rand = tf.random.uniform(shape=tf.shape(images)[0], seed=self.seed)
        is_used = rand < self.probability

        final_images = tf.where(is_used, transformed_images, images)
        final_bboxes_ragged_list = [
            transformed_bboxes_ragged[i] if is_used[i] else bboxes_ragged[i]
            for i in range(len(bboxes_ragged))
        ]

        return final_images, tf.ragged.stack(final_bboxes_ragged_list)


class TFRandomFlipLeftRight(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_flip_left_right, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return super().call(images, bboxes_ragged)


class TFRandomFlipUpDown(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_flip_up_down, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return super().call(images, bboxes_ragged)


class TFRandomRotate90(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_rotate_90, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return super().call(images, bboxes_ragged)


class TFRandomRotate90AndResize(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(_tf_rotate_90_and_pad_and_resize, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return super().call(images, bboxes_ragged)


class TFRandomCropAndResize(TFRandomTransform):

    def __init__(
            self,
            height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(_tf_crop_and_resize, probability, seed)
        self.height_fraction_range = height_fraction_range
        self.width_fraction_range = width_fraction_range

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        images_shape = tf.shape(images)

        def rand_fn() -> tf.Tensor:
            return tf.random.uniform(images_shape[0], seed=self.seed)

        offset_heights, offset_widths, cropped_heights, cropped_widths = \
            _tf_fractions_to_heights_and_widths(
                images_shape[1], images_shape[2],
                self.height_fraction_range, self.width_fraction_range, rand_fn
            )

        return super().call(
            images, bboxes_ragged,
            offset_heights, offset_widths, cropped_heights, cropped_widths
        )
