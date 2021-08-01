"""
API for TensorFlow usage.
"""

from typing import Any, Callable, Sequence, Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

from targetran._functional import (
    _tf_convert,
    _tf_round_to_int,
    _tf_resize_image,
    _tf_pad_image,
    _tf_gather_image
)


from targetran._transform import (
    _flip_left_right,
    _flip_up_down,
    _rotate_90,
    _rotate_90_and_pad,
    _rotate,
    _shear,
    _crop,
    _resize,
    _translate,
    _get_random_crop_inputs,
    _get_random_size_fractions
)


def np_to_tf(
        image_list: Sequence[np.ndarray],
        bboxes_list: Sequence[np.ndarray],
        labels_list: Sequence[np.ndarray]
) -> Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor], Sequence[tf.Tensor]]:
    """
    Convert Numpy array lists to TF (eager) tensor lists.
    """
    tuples = [
        (tf.convert_to_tensor(image, dtype=tf.float32),
         tf.convert_to_tensor(bboxes, dtype=tf.float32),
         tf.convert_to_tensor(labels, dtype=tf.float32))
        for image, bboxes, labels in zip(image_list, bboxes_list, labels_list)
    ]
    tf_image_list, tf_bboxes_list, tf_labels_list = list(zip(*tuples))
    return tf_image_list, tf_bboxes_list, tf_labels_list


def tf_flip_left_right(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _flip_left_right(
        image, bboxes, labels,
        tf.shape, _tf_convert, tf.concat
    )


def tf_flip_up_down(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _flip_up_down(
        image, bboxes, labels,
        tf.shape, _tf_convert, tf.concat
    )


def tf_resize(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        dest_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _resize(
        image, bboxes, labels,
        dest_size, tf.shape, _tf_resize_image, _tf_convert, tf.concat
    )


def tf_rotate_90(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _rotate_90(
        image, bboxes, labels,
        tf.shape, _tf_convert, tf.transpose, tf.concat
    )


def tf_rotate_90_and_pad(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _rotate_90_and_pad(
        image, bboxes, labels,
        tf.shape, _tf_convert, tf.transpose, tf.concat,
        tf.where, tf.math.ceil, tf.math.floor, _tf_pad_image
    )


def tf_rotate(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        angle_deg: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _rotate(
        image, bboxes, labels, angle_deg,
        tf.shape, _tf_convert, tf.expand_dims, tf.squeeze,
        _tf_pad_image, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.stack, tf.concat, tf.cos, tf.sin, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.reshape, tf.identity,
        tf.reduce_max, tf.reduce_min, tf.logical_and, tf.boolean_mask
    )


def tf_shear(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        angle_deg: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _shear(
        image, bboxes, labels, angle_deg,
        tf.shape, _tf_convert, tf.expand_dims, tf.squeeze,
        _tf_pad_image, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.stack, tf.concat, tf.tan, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.reshape, tf.identity,
        tf.reduce_max, tf.reduce_min, tf.logical_and, tf.boolean_mask
    )


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


def tf_crop(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        offset_height: int,
        offset_width: int,
        cropped_image_height: int,
        cropped_image_width: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _crop(
        image, bboxes, labels,
        offset_height, offset_width,
        cropped_image_height, cropped_image_width,
        tf.shape, tf.reshape, _tf_convert, tf.concat,
        tf.logical_and, tf.squeeze, tf.boolean_mask
    )


def tf_translate(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        translate_height: int,
        translate_width: int
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _translate(
        image, bboxes, labels,
        translate_height, translate_width,
        tf.shape, tf.reshape, _tf_convert, tf.where, tf.abs, tf.concat,
        tf.logical_and, tf.squeeze, tf.boolean_mask, _tf_pad_image
    )


class TFResize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return tf_resize(image, bboxes, labels, self.dest_size)


class TFRandomTransform:

    def __init__(
            self,
            tf_fn: Callable[..., Tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
            probability: float,
            seed: int,
    ) -> None:
        self._tf_fn = tf_fn
        self.probability = probability
        self._rng = tf.random.Generator.from_seed(seed)
        self._rand_fn: Callable[..., tf.Tensor] = lambda: self._rng.uniform([])

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        if self._rand_fn() < self.probability:
            return self._tf_fn(image, bboxes, labels, *args, **kwargs)
        return image, bboxes, labels


class TFRandomFlipLeftRight(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(tf_flip_left_right, probability, seed)

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return super().__call__(image, bboxes, labels)


class TFRandomFlipUpDown(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(tf_flip_up_down, probability, seed)

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return super().__call__(image, bboxes, labels)


class TFRandomRotate90(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(tf_rotate_90, probability, seed)

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return super().__call__(image, bboxes, labels)


class TFRandomRotate90AndPad(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(tf_rotate_90_and_pad, probability, seed)

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return super().__call__(image, bboxes, labels)


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
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        angle_deg = \
            _tf_convert(self.angle_deg_range[1] - self.angle_deg_range[0]) \
            * self._rand_fn() + _tf_convert(self.angle_deg_range[0])

        return super().__call__(image, bboxes, labels, angle_deg)


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
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        angle_deg = \
            _tf_convert(self.angle_deg_range[1] - self.angle_deg_range[0]) \
            * self._rand_fn() + _tf_convert(self.angle_deg_range[0])

        return super().__call__(image, bboxes, labels, angle_deg)


class TFRandomCrop(TFRandomTransform):

    def __init__(
            self,
            crop_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(tf_crop, probability, seed)
        self.crop_height_fraction_range = crop_height_fraction_range
        self.crop_width_fraction_range = crop_width_fraction_range

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        offset_height, offset_width, cropped_height, cropped_width = \
            _tf_get_random_crop_inputs(
                tf.shape(image)[0], tf.shape(image)[1],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                self._rand_fn
            )

        return super().__call__(
            image, bboxes, labels,
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
        super().__init__(tf_translate, probability, seed)
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        height_fraction, width_fraction = _get_random_size_fractions(
            self.translate_height_fraction_range,
            self.translate_width_fraction_range,
            self._rand_fn, _tf_convert
        )

        translate_height = tf.shape(image)[0] * height_fraction
        translate_width = tf.shape(image)[1] * width_fraction

        return super().__call__(
            image, bboxes, labels, translate_height, translate_width
        )
