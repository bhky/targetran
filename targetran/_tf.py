"""
API for TensorFlow usage.
"""

from typing import Any, Callable, Tuple

import tensorflow as tf  # type: ignore

from ._functional import _tf_convert
from ._transform import (
    _get_random_size_fractions,
    _tf_resize,
    _tf_flip_left_right,
    _tf_flip_up_down,
    _tf_rotate_90,
    _tf_rotate_90_and_pad_and_resize,
    _tf_get_random_crop_inputs,
    _tf_crop_and_resize,
    _tf_translate
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
            crop_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(_tf_crop_and_resize, probability, seed)
        self.crop_height_fraction_range = crop_height_fraction_range
        self.crop_width_fraction_range = crop_width_fraction_range

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        images_shape = tf.shape(images)

        def rand_fn() -> tf.Tensor:
            return tf.random.uniform(images_shape[0], seed=self.seed)

        offset_heights, offset_widths, cropped_heights, cropped_widths = \
            _tf_get_random_crop_inputs(
                images_shape[1], images_shape[2],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                rand_fn
            )

        return super().call(
            images, bboxes_ragged,
            offset_heights, offset_widths, cropped_heights, cropped_widths
        )


class RandomTranslate(TFRandomTransform):

    def __init__(
            self,
            translate_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            translate_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(_tf_translate, probability, seed)
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        images_shape = tf.shape(images)

        def rand_fn() -> tf.Tensor:
            return tf.random.uniform(images_shape[0], seed=self.seed)

        height_fractions, width_fractions = _get_random_size_fractions(
            self.translate_height_fraction_range,
            self.translate_width_fraction_range,
            rand_fn, _tf_convert
        )

        translate_heights = images_shape[1] * height_fractions
        translate_widths = images_shape[2] * width_fractions

        return super().call(
            images, bboxes_ragged, translate_heights, translate_widths
        )
