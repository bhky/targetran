"""
API for TensorFlow usage.
"""

from typing import Any, Callable, Optional, Sequence, Tuple

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
        image_seq: Sequence[np.ndarray],
        bboxes_seq: Sequence[np.ndarray],
        labels_seq: Sequence[np.ndarray]
) -> Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor], Sequence[tf.Tensor]]:
    """
    Convert Numpy array seqs to TF (eager) tensor seqs.
    """
    tuples = [
        (_tf_convert(image),
         tf.reshape(_tf_convert(bboxes), (-1, 4)),
         _tf_convert(labels))
        for image, bboxes, labels in zip(image_seq, bboxes_seq, labels_seq)
    ]
    tf_image_seq, tf_bboxes_seq, tf_labels_seq = tuple(zip(*tuples))
    return tf_image_seq, tf_bboxes_seq, tf_labels_seq


def seqs_to_tf_dataset(
        image_seq: Sequence[np.ndarray],
        bboxes_seq: Sequence[np.ndarray],
        labels_seq: Sequence[np.ndarray]
) -> tf.data.Dataset:

    tf_image_seq, tf_bboxes_seq, tf_labels_seq = np_to_tf(
        image_seq, bboxes_seq, labels_seq
    )

    # Tensors of different shapes can be included in a TF Dataset
    # as ragged-tensors.
    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_image_seq)),
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_bboxes_seq)),
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_labels_seq))
    ))
    # However, our transformations expect normal tensors, so the ragged-tensors
    # have to be first converted back to tensors during mapping. Therefore,
    # the whole point of using ragged-tensors is ONLY for building a Dataset...
    # Note that the label ragged-tensors are of rank-0, so they are implicitly
    # converted to tensors during mapping. Strange TF Dataset behaviour...
    ds = ds.map(lambda i, b, l: (i.to_tensor(), b.to_tensor(), l))
    return ds


def tf_flip_left_right(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _flip_left_right(
        image, bboxes, labels,
        _tf_convert, tf.shape, tf.reshape, tf.concat
    )


def tf_flip_up_down(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _flip_up_down(
        image, bboxes, labels,
        _tf_convert, tf.shape, tf.reshape, tf.concat
    )


def tf_resize(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        dest_size: Tuple[int, int]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _resize(
        image, bboxes, labels, dest_size,
        _tf_convert, tf.shape, tf.reshape, _tf_resize_image, tf.concat
    )


def tf_rotate_90(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _rotate_90(
        image, bboxes, labels,
        _tf_convert, tf.shape, tf.reshape, tf.transpose, tf.concat
    )


def tf_rotate_90_and_pad(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _rotate_90_and_pad(
        image, bboxes, labels,
        _tf_convert, tf.shape, tf.reshape, tf.transpose, tf.concat,
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
        _tf_convert, tf.shape, tf.reshape, tf.expand_dims, tf.squeeze,
        _tf_pad_image, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.stack, tf.concat, tf.cos, tf.sin, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.identity, tf.reduce_max, tf.reduce_min,
        tf.logical_and, tf.boolean_mask
    )


def tf_shear(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        angle_deg: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _shear(
        image, bboxes, labels, angle_deg,
        _tf_convert, tf.shape, tf.reshape, tf.expand_dims, tf.squeeze,
        _tf_pad_image, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.stack, tf.concat, tf.tan, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.identity, tf.reduce_max, tf.reduce_min,
        tf.logical_and, tf.boolean_mask
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
        _tf_convert, tf.shape, tf.reshape, tf.concat,
        tf.logical_and, tf.squeeze, tf.clip_by_value, tf.boolean_mask
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
        _tf_convert, tf.shape, tf.reshape, tf.where, tf.abs, tf.concat,
        tf.logical_and, tf.squeeze, tf.clip_by_value, tf.boolean_mask,
        _tf_pad_image
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
            seed: Optional[int],
    ) -> None:
        self._tf_fn = tf_fn
        self.probability = probability
        self._rng = tf.random.Generator.from_seed(seed) if seed is not None \
            else tf.random.Generator.from_non_deterministic_state()
        self._rand_fn: Callable[..., tf.Tensor] = lambda: self._rng.uniform([])

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        image = _tf_convert(image)
        bboxes = _tf_convert(bboxes)
        labels = _tf_convert(labels)

        if self._rand_fn() < self.probability:
            return self._tf_fn(image, bboxes, labels, *args, **kwargs)
        return image, bboxes, labels


class TFRandomFlipLeftRight(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.7,
            seed: Optional[int] = None
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
            probability: float = 0.7,
            seed: Optional[int] = None
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
            probability: float = 0.7,
            seed: Optional[int] = None
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
            probability: float = 0.7,
            seed: Optional[int] = None
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
            probability: float = 0.7,
            seed: Optional[int] = None
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
            probability: float = 0.7,
            seed: Optional[int] = None
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
            crop_height_fraction_range: Tuple[float, float] = (0.7, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.7, 0.9),
            probability: float = 0.7,
            seed: Optional[int] = None
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
            translate_height_fraction_range: Tuple[float, float] = (-0.2, 0.2),
            translate_width_fraction_range: Tuple[float, float] = (-0.2, 0.2),
            probability: float = 0.7,
            seed: Optional[int] = None
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

        translate_height = _tf_round_to_int(
            _tf_convert(tf.shape(image)[0]) * height_fraction
        )
        translate_width = _tf_round_to_int(
            _tf_convert(tf.shape(image)[1]) * width_fraction
        )

        return super().__call__(
            image, bboxes, labels, translate_height, translate_width
        )
