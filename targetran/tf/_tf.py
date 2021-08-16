"""
API for TensorFlow usage.
"""

from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar

import functools
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
    _affine_transform,
    _flip_left_right,
    _flip_up_down,
    _rotate,
    _shear,
    _translate,
    _get_random_crop_inputs,
    _get_random_size_fractions,
    _crop,
    _resize,
    _get_flip_left_right_mats,
    _get_flip_up_down_mats,
    _get_rotate_mats,
    _get_shear_mats,
    _get_translate_mats
)

T = TypeVar("T", np.ndarray, tf.Tensor)


def to_tf(
        image_seq: Sequence[T],
        bboxes_seq: Sequence[T],
        labels_seq: Sequence[T]
) -> Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor], Sequence[tf.Tensor]]:
    """
    Convert seqs to TF (eager) tensor seqs.
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
        image_seq: Sequence[T],
        bboxes_seq: Sequence[T],
        labels_seq: Sequence[T]
) -> tf.data.Dataset:
    tf_image_seq, tf_bboxes_seq, tf_labels_seq = to_tf(
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


def _tf_affine_transform(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        image_dest_tran_mat: tf.Tensor,
        bboxes_tran_mat: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _affine_transform(
        image, bboxes, labels,
        _tf_convert, tf.shape, tf.reshape, tf.expand_dims, tf.squeeze,
        _tf_pad_image, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.ones_like, tf.stack, tf.concat,
        image_dest_tran_mat, bboxes_tran_mat, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.identity, tf.reduce_max, tf.reduce_min,
        tf.logical_and, tf.boolean_mask
    )


def tf_flip_left_right(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _flip_left_right(
        image, bboxes, labels,
        _tf_convert, tf.shape, tf.reshape, tf.expand_dims, tf.squeeze,
        _tf_pad_image, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.ones_like, tf.stack, tf.concat, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.identity, tf.reduce_max, tf.reduce_min,
        tf.logical_and, tf.boolean_mask
    )


def tf_flip_up_down(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _flip_up_down(
        image, bboxes, labels,
        _tf_convert, tf.shape, tf.reshape, tf.expand_dims, tf.squeeze,
        _tf_pad_image, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.ones_like, tf.stack, tf.concat, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.identity, tf.reduce_max, tf.reduce_min,
        tf.logical_and, tf.boolean_mask
    )


def tf_rotate(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        angle_deg: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _rotate(
        image, bboxes, labels, _tf_convert(angle_deg),
        _tf_convert, tf.cos, tf.sin, tf.shape, tf.reshape,
        tf.expand_dims, tf.squeeze, _tf_pad_image, tf.range,
        _tf_round_to_int, tf.repeat, tf.tile,
        tf.ones_like, tf.stack, tf.concat, tf.matmul,
        tf.clip_by_value, _tf_gather_image, tf.identity,
        tf.reduce_max, tf.reduce_min, tf.logical_and, tf.boolean_mask
    )


def tf_shear(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        angle_deg: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _shear(
        image, bboxes, labels, _tf_convert(angle_deg),
        _tf_convert, tf.tan, tf.shape, tf.reshape, tf.expand_dims, tf.squeeze,
        _tf_pad_image, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.ones_like, tf.stack, tf.concat, tf.matmul, tf.clip_by_value,
        _tf_gather_image, tf.identity, tf.reduce_max, tf.reduce_min,
        tf.logical_and, tf.boolean_mask
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
        _tf_convert(translate_height), _tf_convert(translate_width),
        _tf_convert, tf.shape, tf.reshape, tf.expand_dims, tf.squeeze,
        _tf_pad_image, tf.range, _tf_round_to_int, tf.repeat, tf.tile,
        tf.ones_like, tf.stack, tf.concat, tf.matmul, tf.clip_by_value,
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
        _tf_convert(offset_height), _tf_convert(offset_width),
        _tf_convert(cropped_image_height), _tf_convert(cropped_image_width),
        _tf_convert, tf.shape, tf.reshape, tf.concat,
        tf.logical_and, tf.squeeze, tf.clip_by_value, tf.boolean_mask
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

    def get_mats(self, image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Make sure inputs are in the needed format.
        image = _tf_convert(image)
        bboxes = _tf_convert(bboxes)
        labels = _tf_convert(labels)

        if self._rand_fn() < self.probability:
            return self._tf_fn(image, bboxes, labels, *args, **kwargs)
        return image, bboxes, labels


class TFCombineAffine(TFRandomTransform):

    def __init__(
            self,
            transforms: Sequence[TFRandomTransform],
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        self._transforms = transforms
        super().__init__(_tf_affine_transform, probability, seed)

    def _combine_mats(self, image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image_dest_tran_mats, bboxes_tran_mats = tuple(zip(
            *[tran.get_mats(image) for tran in self._transforms]
        ))
        image_dest_tran_mat = functools.reduce(tf.matmul, image_dest_tran_mats)
        bboxes_tran_mat = functools.reduce(tf.matmul, bboxes_tran_mats)
        return image_dest_tran_mat, bboxes_tran_mat

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        image_dest_tran_mat, bboxes_tran_mat = self._combine_mats(image)
        return super().__call__(
            image, bboxes, labels, image_dest_tran_mat, bboxes_tran_mat
        )


class TFRandomFlipLeftRight(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.7,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(tf_flip_left_right, probability, seed)

    def get_mats(self, image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return _get_flip_left_right_mats(_tf_convert)

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

    def get_mats(self, image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return _get_flip_up_down_mats(_tf_convert)

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

    def _get_angle_deg(self) -> tf.Tensor:
        return _tf_convert(self.angle_deg_range[1] - self.angle_deg_range[0]) \
               * self._rand_fn() + _tf_convert(self.angle_deg_range[0])

    def get_mats(self, image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return _get_rotate_mats(
            self._get_angle_deg(), _tf_convert, tf.cos, tf.sin
        )

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

    def _get_angle_deg(self) -> tf.Tensor:
        return _tf_convert(self.angle_deg_range[1] - self.angle_deg_range[0]) \
               * self._rand_fn() + _tf_convert(self.angle_deg_range[0])

    def get_mats(self, image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return _get_shear_mats(self._get_angle_deg(), _tf_convert, tf.tan)

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return super().__call__(image, bboxes, labels, self._get_angle_deg())


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

    def _get_translate_height_and_width(
            self,
            image: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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
        return translate_height, translate_width

    def get_mats(self, image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        translate_height, translate_width = \
            self._get_translate_height_and_width(image)
        return _get_translate_mats(
            translate_height, translate_width, _tf_convert
        )

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        translate_height, translate_width = \
            self._get_translate_height_and_width(image)
        return super().__call__(
            image, bboxes, labels, translate_height, translate_width
        )


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
