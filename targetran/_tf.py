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
    _tf_round_to_int,
    _tf_resize_image,
    _tf_pad_images,
    _tf_gather_image,
    _tf_make_bboxes_ragged
)


from ._transform import (
    _flip_left_right,
    _flip_up_down,
    _rotate_90,
    _rotate_90_and_pad,
    _rotate_single,
    _shear_single,
    _crop_single,
    _resize_single,
    _translate_single,
    _get_random_crop_inputs,
    _get_random_size_fractions
)


def tf_flip_left_right(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    return _flip_left_right(
        images, bboxes_ragged,
        tf.shape, _tf_convert, _tf_stack_bboxes, tf.concat,
        _tf_make_bboxes_ragged
    )


def tf_flip_up_down(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    return _flip_up_down(
        images, bboxes_ragged,
        tf.shape, _tf_convert, _tf_stack_bboxes, tf.concat,
        _tf_make_bboxes_ragged
    )


def tf_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
        dest_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    image_list = [image for image in images]
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    image_list, bboxes_ragged = _map_single(
        _resize_single, image_list, bboxes_list, None,
        dest_size, tf.shape, _tf_resize_image, _tf_convert, tf.concat
    )
    images = _tf_convert(image_list)
    bboxes_ragged = _tf_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


def tf_rotate_90(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    return _rotate_90(
        images, bboxes_ragged,
        tf.shape, _tf_convert, tf.transpose, _tf_stack_bboxes, tf.concat,
        _tf_make_bboxes_ragged
    )


def _tf_rotate_90_and_pad(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    """
    Middle-step function for easy testing.
    """
    return _rotate_90_and_pad(
        images, bboxes_ragged,
        tf.shape, _tf_convert, tf.transpose, _tf_stack_bboxes, tf.concat,
        tf.where, tf.math.ceil, tf.math.floor, _tf_pad_images,
        _tf_make_bboxes_ragged
    )


def tf_rotate_90_and_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    """
    Could be tf_rotate_90_and_pad_and_resize, but thought it is too clumsy.
    """
    height, width = int(tf.shape(images)[1]), int(tf.shape(images)[2])
    images, bboxes_ragged = _tf_rotate_90_and_pad(images, bboxes_ragged)
    return tf_resize(images, bboxes_ragged, (height, width))


def tf_rotate(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
        angles_deg: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    image_list = [image for image in images]
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _rotate_single, image_list, bboxes_list,
        [angles_deg],
        tf.shape, _tf_convert, tf.expand_dims, tf.squeeze,
        _tf_pad_images, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.stack, tf.concat, tf.cos, tf.sin, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.reshape, tf.identity,
        tf.reduce_max, tf.reduce_min, tf.logical_and, tf.boolean_mask
    )
    images = _tf_convert(image_list)
    bboxes_ragged = _tf_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


def tf_shear(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
        angles_deg: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
    image_list = [image for image in images]
    bboxes_list = _tf_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _shear_single, image_list, bboxes_list,
        [angles_deg],
        tf.shape, _tf_convert, tf.expand_dims, tf.squeeze,
        _tf_pad_images, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.stack, tf.concat, tf.tan, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.reshape, tf.identity,
        tf.reduce_max, tf.reduce_min, tf.logical_and, tf.boolean_mask
    )
    images = _tf_convert(image_list)
    bboxes_ragged = _tf_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


def _tf_get_random_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return _get_random_crop_inputs(
        image_height, image_width, height_fraction_range, width_fraction_range,
        rand_fn, _tf_convert, _tf_round_to_int
    )


def tf_crop_and_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
        offset_heights: tf.Tensor,
        offset_widths: tf.Tensor,
        cropped_image_heights: tf.Tensor,
        cropped_image_widths: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
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
        bboxes_ragged: tf.RaggedTensor,
        translate_heights: tf.Tensor,
        translate_widths: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:
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
            bboxes_ragged: tf.RaggedTensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        return tf_resize(images, bboxes_ragged, self.dest_size)


class TFRandomTransform:

    def __init__(
            self,
            tf_fn: Callable[..., Tuple[tf.Tensor, tf.RaggedTensor]],
            probability: float,
            seed: int,
    ) -> None:
        self._tf_fn = tf_fn
        self.probability = probability
        self.seed = seed

    def call(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.RaggedTensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:

        transformed_images, transformed_bboxes_ragged = self._tf_fn(
            images, bboxes_ragged, *args, **kwargs
        )

        rand = tf.random.uniform(shape=tf.shape(images)[0], seed=self.seed)
        is_used = rand < self.probability

        final_images = tf.where(is_used, transformed_images, images)
        final_bboxes_ragged_list = [
            transformed_bboxes_ragged[i] if is_used[i] else bboxes_ragged[i]
            for i in range(len(images))
        ]

        return final_images, tf.ragged.stack(final_bboxes_ragged_list)


class TFRandomFlipLeftRight(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(tf_flip_left_right, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.RaggedTensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        return super().call(images, bboxes_ragged)


class TFRandomFlipUpDown(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(tf_flip_up_down, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.RaggedTensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        return super().call(images, bboxes_ragged)


class TFRandomRotate90(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(tf_rotate_90, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.RaggedTensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        return super().call(images, bboxes_ragged)


class TFRandomRotate90AndResize(TFRandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(tf_rotate_90_and_resize, probability, seed)

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.RaggedTensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        return super().call(images, bboxes_ragged)


class TFRandomRotate(TFRandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(tf_rotate, probability, seed)
        assert angle_deg_range[0] < angle_deg_range[1]
        self.angle_deg_range = angle_deg_range

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.RaggedTensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:

        images_shape = tf.shape(images)

        def rand_fn() -> tf.Tensor:
            return tf.random.uniform(images_shape[0], seed=self.seed)

        angles_deg = \
            _tf_convert(self.angle_deg_range[1] - self.angle_deg_range[0]) \
            * rand_fn() + _tf_convert(self.angle_deg_range[0])

        return super().call(images, bboxes_ragged, angles_deg)


class TFRandomShear(TFRandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(tf_shear, probability, seed)
        assert -90.0 < angle_deg_range[0] < angle_deg_range[1] < 90.0
        self.angle_deg_range = angle_deg_range

    def __call__(
            self,
            images: tf.Tensor,
            bboxes_ragged: tf.RaggedTensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:

        images_shape = tf.shape(images)

        def rand_fn() -> tf.Tensor:
            return tf.random.uniform(images_shape[0], seed=self.seed)

        angles_deg = \
            _tf_convert(self.angle_deg_range[1] - self.angle_deg_range[0]) \
            * rand_fn() + _tf_convert(self.angle_deg_range[0])

        return super().call(images, bboxes_ragged, angles_deg)


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
            bboxes_ragged: tf.RaggedTensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:

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


class TFRandomTranslate(TFRandomTransform):

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
            bboxes_ragged: tf.RaggedTensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:

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
