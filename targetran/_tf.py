"""
API for TensorFlow usage.
"""

from typing import Any, Callable, Tuple

import tensorflow as tf  # type: ignore

from ._functional import (
    _map_single,
    _tf_convert,
    _tf_ragged_to_list,
    _tf_list_to_ragged,
    _tf_stack_bboxes,
    _tf_resize_image,
    _tf_pad_images,
    _tf_make_bboxes_ragged
)


from ._transform import (
    _flip_left_right,
    _flip_up_down,
    _rotate_90,
    _rotate_90_and_pad,
    _crop_single,
    _resize_single,
    _translate_single,
    _get_random_crop_inputs,
    _get_random_size_fractions
)


def tf_flip_left_right(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _flip_left_right(
        images, bboxes_ragged,
        tf.shape, _tf_convert, _tf_stack_bboxes, tf.concat,
        _tf_make_bboxes_ragged
    )


def tf_flip_up_down(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _flip_up_down(
        images, bboxes_ragged,
        tf.shape, _tf_convert, _tf_stack_bboxes, tf.concat,
        _tf_make_bboxes_ragged
    )


def tf_rotate_90(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _rotate_90(
        images, bboxes_ragged,
        tf.shape, _tf_convert, tf.transpose, _tf_stack_bboxes, tf.concat,
        _tf_make_bboxes_ragged
    )


def tf_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
        dest_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor]:
    image_list = [image for image in images]
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    image_list, bboxes_ragged = _map_single(
        _resize_single, image_list, bboxes_list, None,
        dest_size, tf.shape, _tf_resize_image, _tf_convert, tf.concat
    )
    images = _tf_convert(image_list)
    bboxes_ragged = _tf_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


def tf_rotate_90_and_pad(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _rotate_90_and_pad(
        images, bboxes_ragged,
        tf.shape, _tf_convert, tf.transpose, _tf_stack_bboxes, tf.concat,
        tf.where, tf.math.ceil, tf.math.floor, _tf_pad_images,
        _tf_make_bboxes_ragged
    )


def tf_rotate_90_and_pad_and_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    height, width = int(tf.shape(images)[1]), int(tf.shape(images)[2])
    images, bboxes_ragged = tf_rotate_90_and_pad(images, bboxes_ragged)
    return tf_resize(images, bboxes_ragged, (height, width))


def _tf_get_random_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return _get_random_crop_inputs(
        image_height, image_width, height_fraction_range, width_fraction_range,
        rand_fn, _tf_convert, tf.math.rint
    )


def tf_crop_and_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
        offset_heights: tf.Tensor,
        offset_widths: tf.Tensor,
        cropped_image_heights: tf.Tensor,
        cropped_image_widths: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    image_list = [image for image in images]
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _crop_single, image_list, bboxes_list,
        [offset_heights, offset_widths,
         cropped_image_heights, cropped_image_widths],
        tf.shape, tf.reshape, _tf_convert, tf.concat,
        tf.logical_and, tf.squeeze, tf.boolean_mask
    )
    image_list, bboxes_list = _map_single(
        _resize_single, image_list, bboxes_list, None,
        tf.shape(images)[1:3], tf.shape, _tf_resize_image,
        _tf_convert, tf.concat
    )
    images = _tf_convert(image_list)
    bboxes_ragged = _tf_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


def tf_translate(
        images: tf.Tensor,
        bboxes_ragged: tf.Tensor,
        translate_heights: tf.Tensor,
        translate_widths: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    image_list = [image for image in images]
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _translate_single, image_list, bboxes_list,
        [translate_heights, translate_widths],
        tf.shape, tf.reshape, _tf_convert, tf.where, tf.abs, tf.concat,
        tf.logical_and, tf.expand_dims, tf.squeeze, tf.boolean_mask,
        _tf_pad_images
    )
    images = _tf_convert(image_list)
    bboxes_ragged = _tf_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


class TFResize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf_resize(images, bboxes_ragged, self.dest_size)


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
        super().__init__(tf_flip_left_right, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return super().call(images, bboxes_ragged)


class TFRandomFlipUpDown(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(tf_flip_up_down, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return super().call(images, bboxes_ragged)


class TFRandomRotate90(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(tf_rotate_90, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return super().call(images, bboxes_ragged)


class TFRandomRotate90AndResize(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(tf_rotate_90_and_pad_and_resize, probability, seed)

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
        super().__init__(tf_crop_and_resize, probability, seed)
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
        super().__init__(tf_translate, probability, seed)
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
