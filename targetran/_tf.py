"""
API for TensorFlow usage.
"""

from typing import Any, Callable, Tuple

import tensorflow as tf  # type: ignore

from ._functional import (
    _tf_map_idx_fn,
    _tf_to_single_fn,
    _tf_convert,
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


def _tf_resize_single(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        dest_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _resize_single(
        image, bboxes,
        dest_size, tf.shape, _tf_resize_image, _tf_convert, tf.concat
    )


def tf_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
        dest_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.RaggedTensor]:

    def fn(idx: tf.Tensor) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        image, bboxes = _tf_resize_single(
            images[idx], bboxes_ragged[idx].to_tensor(), dest_size
        )
        return image, tf.RaggedTensor.from_tensor(bboxes)

    return _tf_map_idx_fn(fn, int(tf.shape(images)[0]))


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


def _tf_rotate_single(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        angle_deg: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _rotate_single(
        image, bboxes, angle_deg,
        tf.shape, _tf_convert, tf.expand_dims, tf.squeeze,
        _tf_pad_images, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.stack, tf.concat, tf.cos, tf.sin, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.reshape, tf.identity,
        tf.reduce_max, tf.reduce_min, tf.logical_and, tf.boolean_mask
    )


def tf_rotate(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
        angles_deg: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:

    def fn(idx: tf.Tensor) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        image, bboxes = _tf_rotate_single(
            images[idx], bboxes_ragged[idx].to_tensor(), angles_deg[idx]
        )
        return image, tf.RaggedTensor.from_tensor(bboxes)

    return _tf_map_idx_fn(fn, int(tf.shape(images)[0]))


def _tf_shear_single(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        angle_deg: float
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _shear_single(
        image, bboxes, angle_deg,
        tf.shape, _tf_convert, tf.expand_dims, tf.squeeze,
        _tf_pad_images, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.stack, tf.concat, tf.tan, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.reshape, tf.identity,
        tf.reduce_max, tf.reduce_min, tf.logical_and, tf.boolean_mask
    )


def tf_shear(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
        angles_deg: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:

    def fn(idx: tf.Tensor) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        image, bboxes = _tf_shear_single(
            images[idx], bboxes_ragged[idx].to_tensor(), angles_deg[idx],
        )
        return image, tf.RaggedTensor.from_tensor(bboxes)

    return _tf_map_idx_fn(fn, int(tf.shape(images)[0]))


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


def _tf_crop_and_resize_single(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        offset_height: int,
        offset_width: int,
        cropped_image_height: int,
        cropped_image_width: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    cropped_image, cropped_bboxes = _crop_single(
        image, bboxes,
        offset_height, offset_width,
        cropped_image_height, cropped_image_width,
        tf.shape, tf.reshape, _tf_convert, tf.concat,
        tf.logical_and, tf.squeeze, tf.boolean_mask
    )
    return _tf_resize_single(
        cropped_image, cropped_bboxes, tf.shape(image)[0:2]
    )


def tf_crop_and_resize(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
        offset_heights: tf.Tensor,
        offset_widths: tf.Tensor,
        cropped_image_heights: tf.Tensor,
        cropped_image_widths: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:

    def fn(idx: tf.Tensor) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        image, bboxes = _tf_crop_and_resize_single(
            images[idx],
            bboxes_ragged[idx].to_tensor(),
            offset_heights[idx], offset_widths[idx],
            cropped_image_heights[idx], cropped_image_widths[idx]
        )
        return image, tf.RaggedTensor.from_tensor(bboxes)

    return _tf_map_idx_fn(fn, int(tf.shape(images)[0]))


def _tf_translate_single(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        translate_height: int,
        translate_width: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    return _translate_single(
        image, bboxes,
        translate_height, translate_width,
        tf.shape, tf.reshape, _tf_convert, tf.where, tf.abs, tf.concat,
        tf.logical_and, tf.expand_dims, tf.squeeze, tf.boolean_mask,
        _tf_pad_images
    )


def tf_translate(
        images: tf.Tensor,
        bboxes_ragged: tf.RaggedTensor,
        translate_heights: tf.Tensor,
        translate_widths: tf.Tensor
) -> Tuple[tf.Tensor, tf.RaggedTensor]:

    def fn(idx: tf.Tensor) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        image, bboxes = _tf_translate_single(
            images[idx],
            bboxes_ragged[idx].to_tensor(),
            translate_heights[idx], translate_widths[idx]
        )
        return image, tf.RaggedTensor.from_tensor(bboxes)

    return _tf_map_idx_fn(fn, int(tf.shape(images)[0]))


class TFResize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        new_image, new_bboxes = _tf_resize_single(image, bboxes, self.dest_size)
        return new_image, tf.RaggedTensor.from_tensor(new_bboxes)


class TFRandomTransform:

    def __init__(
            self,
            tf_single_fn: Callable[..., Tuple[tf.Tensor, tf.Tensor]],
            probability: float,
            seed: int,
    ) -> None:
        self._tf_single_fn = tf_single_fn
        self.probability = probability
        self._rand_fn: Callable[..., tf.Tensor] = \
            lambda: tf.random.uniform(shape=[1], seed=seed)[0]

    def call(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        """
        Note: when looping a Dataset (without batching), each row of the
        tf.RaggedTensor (bboxes_ragged) is a bboxes (tf.Tensor). However,
        we still want to make the single output a tf.RaggedTensor
        for consistency, and later batching.
        """
        if self._rand_fn() < self.probability:
            new_image, new_bboxes = self._tf_single_fn(
                image, bboxes, *args, **kwargs
            )
            return new_image, tf.RaggedTensor.from_tensor(new_bboxes)
        return image, tf.RaggedTensor.from_tensor(bboxes)


class TFRandomFlipLeftRight(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(
            _tf_to_single_fn(tf_flip_left_right), probability, seed
        )

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        return super().call(image, bboxes)


class TFRandomFlipUpDown(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(
            _tf_to_single_fn(tf_flip_up_down), probability, seed
        )

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        return super().call(image, bboxes)


class TFRandomRotate90(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(
            _tf_to_single_fn(tf_rotate_90), probability, seed
        )

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        return super().call(image, bboxes)


class TFRandomRotate90AndResize(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(
            _tf_to_single_fn(tf_rotate_90_and_resize), probability, seed
        )

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:
        return super().call(image, bboxes)


class TFRandomRotate(TFRandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(_tf_rotate_single, probability, seed)
        assert angle_deg_range[0] < angle_deg_range[1]
        self.angle_deg_range = angle_deg_range

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:

        angle_deg = \
            _tf_convert(self.angle_deg_range[1] - self.angle_deg_range[0]) \
            * self._rand_fn() + _tf_convert(self.angle_deg_range[0])

        return super().call(image, bboxes, angle_deg)


class TFRandomShear(TFRandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-15.0, 15.0),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(_tf_shear_single, probability, seed)
        assert -90.0 < angle_deg_range[0] < angle_deg_range[1] < 90.0
        self.angle_deg_range = angle_deg_range

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:

        angle_deg = \
            _tf_convert(self.angle_deg_range[1] - self.angle_deg_range[0]) \
            * self._rand_fn() + _tf_convert(self.angle_deg_range[0])

        return super().call(image, bboxes, angle_deg)


class TFRandomCropAndResize(TFRandomTransform):

    def __init__(
            self,
            crop_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(_tf_crop_and_resize_single, probability, seed)
        self.crop_height_fraction_range = crop_height_fraction_range
        self.crop_width_fraction_range = crop_width_fraction_range

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:

        offset_height, offset_width, cropped_height, cropped_width = \
            _tf_get_random_crop_inputs(
                tf.shape(image)[0], tf.shape(image)[1],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                self._rand_fn
            )

        return super().call(
            image, bboxes,
            offset_height, offset_width, cropped_height, cropped_width
        )


class TFRandomTranslate(TFRandomTransform):

    def __init__(
            self,
            translate_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            translate_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(_tf_translate_single, probability, seed)
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.RaggedTensor]:

        height_fraction, width_fraction = _get_random_size_fractions(
            self.translate_height_fraction_range,
            self.translate_width_fraction_range,
            self._rand_fn, _tf_convert
        )

        translate_height = tf.shape(image)[0] * height_fraction
        translate_width = tf.shape(image)[1] * width_fraction

        return super().call(
            image, bboxes, translate_height, translate_width
        )
