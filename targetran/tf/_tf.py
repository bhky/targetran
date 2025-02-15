"""
API for TensorFlow usage.
"""
import itertools
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf  # type: ignore

from targetran._check import (
    _check_shear_input,
    _check_translate_input,
    _check_crop_input,
    _check_input_range,
)
from targetran._tf_functional import (
    _tf_convert,
    _tf_cast_to_int,
    _tf_round_to_int,
    _tf_resize_image,
    _tf_pad_image,
    _tf_gather_image,
)
from targetran._transform import (
    _AffineDependency,
    _affine_transform,
    _flip_left_right,
    _flip_up_down,
    _rotate,
    _shear,
    _translate,
    _get_crop_inputs,
    _get_random_size_fractions,
    _crop,
    _resize,
    _get_flip_left_right_mats,
    _get_flip_up_down_mats,
    _get_rotate_mats,
    _get_shear_mats,
    _get_translate_mats,
)
from targetran._typing import T
from targetran.utils import Interpolation


def load_tf_image(image_path: str) -> tf.Tensor:
    return tf.image.decode_image(
        tf.io.read_file(image_path), channels=3, expand_animations=False
    )


def to_tf(
        image_seq: Sequence[T],
        bboxes_seq: Sequence[T] = (),
        labels_seq: Sequence[T] = (),
        image_seq_is_paths: bool = False,
) -> Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor], Sequence[tf.Tensor]]:
    """
    Convert seqs to TF (eager) tensor seqs.
    """
    tuples: List[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]] = []
    for image, bboxes, labels in itertools.zip_longest(  # type: ignore
            image_seq, bboxes_seq, labels_seq, fillvalue=[]
    ):
        tuples.append(
            (image if image_seq_is_paths else _tf_convert(image),
             tf.reshape(_tf_convert(bboxes), (-1, 4)),
             _tf_convert(labels))
        )
    tf_image_seq, tf_bboxes_seq, tf_labels_seq = tuple(zip(*tuples))

    if image_seq_is_paths:
        tf_image_seq = tf.convert_to_tensor(tf_image_seq, tf.string)

    return tf_image_seq, tf_bboxes_seq, tf_labels_seq


def to_tf_dataset(
        image_seq: Sequence[T],
        bboxes_seq: Sequence[T] = (),
        labels_seq: Sequence[T] = (),
        image_seq_is_paths: bool = False,
) -> tf.data.Dataset:
    tf_image_seq, tf_bboxes_seq, tf_labels_seq = to_tf(
        image_seq, bboxes_seq, labels_seq, image_seq_is_paths
    )

    if image_seq_is_paths:
        ds_image = tf.data.Dataset.from_tensor_slices(tf_image_seq)
        ds_image = ds_image.map(load_tf_image)
    else:
        ds_image = tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_image_seq))

    # Tensors of different shapes can be included in a TF Dataset
    # as ragged-tensors.
    ds = tf.data.Dataset.zip((
        ds_image,
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_bboxes_seq)),
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_labels_seq))
    ))
    # However, our transformations expect normal tensors, so the ragged-tensors
    # have to be first converted back to tensors during mapping. Therefore,
    # the whole point of using ragged-tensors is ONLY for building a Dataset...
    # Note that the label ragged-tensors are of rank-0, so they are implicitly
    # converted to tensors during mapping. Strange TF Dataset behaviour...
    ds = ds.map(
        lambda i, b, l: (
            i if isinstance(i, tf.Tensor) else i.to_tensor(),
            b if isinstance(b, tf.Tensor) else b.to_tensor(),
            l
        )
    )
    return ds


def to_keras_cv_dict(
        ds: tf.data.Dataset,
        batch_size: Optional[int] = None,
        drop_remainder: bool = True,
) -> tf.data.Dataset:
    import keras_cv  # type: ignore

    ds = ds.map(lambda i, b, l: (i, {"boxes": b, "classes": l}))
    if batch_size:
        ds = ds.ragged_batch(batch_size=batch_size, drop_remainder=drop_remainder)

    ds = ds.map(lambda i, d: {"images": i, "bounding_boxes": d})
    return ds


def to_keras_cv_model_input(
        ds: tf.data.Dataset,
        max_num_bboxes: Optional[int] = None,
        fill_value: int = -1,
) -> tf.data.Dataset:
    import keras_cv

    return ds.map(
        lambda d: (
            d["images"],
            keras_cv.bounding_box.to_dense(
                d["bounding_boxes"],
                max_boxes=max_num_bboxes,
                default_value=fill_value
            )
        )
    )


def _tf_get_affine_dependency() -> _AffineDependency:
    return _AffineDependency(
        _tf_convert, tf.shape, tf.reshape, tf.expand_dims, tf.squeeze,
        _tf_pad_image, tf.range, _tf_cast_to_int, _tf_round_to_int,
        tf.repeat, tf.tile, tf.ones_like, tf.stack, tf.concat, tf.matmul,
        tf.clip_by_value, tf.math.floor, tf.math.ceil, tf.transpose,
        _tf_gather_image, tf.identity, tf.reduce_max, tf.reduce_min,
        tf.logical_and, tf.boolean_mask
    )


def _tf_affine_transform(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        image_dest_tran_mat: tf.Tensor,
        bboxes_tran_mat: tf.Tensor,
        interpolation: Interpolation,
        fill_value: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _affine_transform(
        image, bboxes, labels, image_dest_tran_mat, bboxes_tran_mat,
        interpolation, fill_value, _tf_get_affine_dependency()
    )


def tf_flip_left_right(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _flip_left_right(
        image, bboxes, labels, _tf_get_affine_dependency()
    )


def tf_flip_up_down(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _flip_up_down(
        image, bboxes, labels, _tf_get_affine_dependency()
    )


def tf_rotate(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        angle_deg: float,
        interpolation: Interpolation = Interpolation.BILINEAR,
        fill_value: float = 0.0
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _rotate(
        image, bboxes, labels, _tf_convert(angle_deg), tf.cos, tf.sin,
        interpolation, fill_value, _tf_get_affine_dependency()
    )


def tf_shear(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        angle_deg: float,
        interpolation: Interpolation = Interpolation.BILINEAR,
        fill_value: float = 0.0,
        _check_input: bool = True
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    if _check_input:
        _check_shear_input(angle_deg)
    return _shear(
        image, bboxes, labels, _tf_convert(angle_deg), tf.tan,
        interpolation, fill_value, _tf_get_affine_dependency()
    )


def tf_translate(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        translate_height: int,
        translate_width: int,
        interpolation: Interpolation = Interpolation.BILINEAR,
        fill_value: float = 0.0,
        _check_input: bool = True
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    if _check_input:
        _check_translate_input(
            image.get_shape(), translate_height, translate_width
        )
    return _translate(
        image, bboxes, labels,
        _tf_convert(translate_height), _tf_convert(translate_width),
        interpolation, fill_value, _tf_get_affine_dependency()
    )


def _tf_get_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return _get_crop_inputs(
        image_height, image_width, height_fraction_range, width_fraction_range,
        rand_fn, _tf_convert, _tf_round_to_int
    )


def tf_crop(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        offset_height: int,
        offset_width: int,
        crop_height: int,
        crop_width: int,
        _check_input: bool = True
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    if _check_input:
        _check_crop_input(image.get_shape(), offset_height, offset_width)
    return _crop(
        image, bboxes, labels,
        _tf_convert(offset_height), _tf_convert(offset_width),
        _tf_convert(crop_height), _tf_convert(crop_width),
        _tf_convert, tf.shape, tf.reshape, tf.concat,
        tf.logical_and, tf.squeeze, tf.clip_by_value, tf.boolean_mask
    )


def tf_resize(
        image: tf.Tensor,
        bboxes: tf.Tensor,
        labels: tf.Tensor,
        dest_size: Tuple[int, int],
        interpolation: Interpolation = Interpolation.BILINEAR
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return _resize(
        image, bboxes, labels, dest_size, interpolation,
        _tf_convert, tf.shape, tf.reshape, _tf_resize_image, tf.concat
    )


class TFRandomTransform:

    def __init__(
            self,
            tf_fn: Callable[..., Tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
            probability: float,
            seed: Optional[int],
            name: str,
            is_affine: bool
    ) -> None:
        self._tf_fn = tf_fn
        self.probability = probability
        self.seed = seed if seed is not None else np.random.randint(1e6)  # type: ignore
        self._rng = tf.random.Generator.from_seed(self.seed)
        self.name = name
        self.is_affine = is_affine

    def _rand_fn(self, shape: Sequence[int] = ()) -> tf.Tensor:
        return self._rng.uniform(shape=shape)

    def _get_mats(  # type: ignore[empty-body]
            self,
            image: tf.Tensor,
            rand_fn: Callable[..., tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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

        if self._rand_fn() < _tf_convert(self.probability):
            return self._tf_fn(image, bboxes, labels, *args, **kwargs)
        return image, bboxes, labels


def _get_random_indices(
        rng: tf.random.Generator,
        num_indices: int,
        num_selected_indices: int,
        selected_probabilities: Optional[List[float]] = None,
) -> tf.Tensor:
    """
    Roughly mimicking Numpy choice for getting indices, without replacement.
    The indices always start from 0.

    References:
    https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
    """
    probs = selected_probabilities if selected_probabilities \
        else tf.ones(num_indices)
    logits = tf.math.log(probs)
    z = -tf.math.log(-tf.math.log(  # pylint: disable=invalid-unary-operand-type
        rng.uniform(tf.shape(logits), 0, 1)
    ))
    _, indices = tf.math.top_k(logits + z, num_selected_indices, sorted=False)
    return indices


class TFCombineAffine(TFRandomTransform):

    def __init__(
            self,
            transforms: Sequence[TFRandomTransform],
            num_selected_transforms: Optional[int] = None,
            selected_probabilities: Optional[List[float]] = None,
            keep_order: bool = False,
            interpolation: Interpolation = Interpolation.BILINEAR,
            fill_value: float = 0.0,
            probability: float = 1.0,
            seed: Optional[int] = None
    ) -> None:
        not_affine_trans = list(filter(lambda t: not t.is_affine, transforms))
        if not_affine_trans:
            raise AssertionError(
                f"Non-affine transforms cannot be included in TFCombineAffine: "
                f"{[t.name for t in not_affine_trans]}"
            )
        if num_selected_transforms and selected_probabilities:
            if len(selected_probabilities) != len(transforms):
                raise ValueError(
                    "Number of items in selected_probabilities should be "
                    "the same as the number of items in transforms."
                )
        super().__init__(
            _tf_affine_transform, probability, seed, "TFCombineAffine", True
        )
        self._transforms = transforms
        self._num_selected_transforms = num_selected_transforms
        self._selected_probabilities = selected_probabilities
        self._keep_order = keep_order
        self._interpolation = interpolation
        self._fill_value = fill_value

    def _get_mats(
            self,
            image: tf.Tensor,
            rand_fn: Callable[..., tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        image_dest_tran_mats, bboxes_tran_mats, probs = tuple(zip(
            *[(*t._get_mats(image, rand_fn), t.probability)
              for i, t in enumerate(self._transforms)]
        ))

        if self._num_selected_transforms is not None:
            if self._num_selected_transforms == 0:
                return tf.eye(3), tf.eye(3)

            indices = _get_random_indices(
                self._rng,
                len(self._transforms),
                self._num_selected_transforms,
                self._selected_probabilities
            )
        else:
            # It looks like in graph mode we cannot check if a tensor is empty
            # and then do short circuit. Therefore, to prevent empty indices
            # which tf.gather could not handle, an identity matrix is appended.
            image_dest_tran_mats += (tf.eye(3),)
            bboxes_tran_mats += (tf.eye(3),)
            probs += (1.0,)

            conditions = rand_fn() < probs
            indices = tf.boolean_mask(
                tf.range(len(probs), dtype=tf.int32), conditions
            )

        if self._keep_order:
            indices = tf.sort(indices)
        else:
            indices = tf.random.shuffle(indices, seed=self.seed)

        image_dest_tran_mats = tf.gather(  # pylint: disable=no-value-for-parameter
            image_dest_tran_mats, indices
        )
        bboxes_tran_mats = tf.gather(  # pylint: disable=no-value-for-parameter
            bboxes_tran_mats, indices
        )

        def matmul(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
            return tf.matmul(a, b)

        image_dest_tran_mat = tf.scan(matmul, image_dest_tran_mats)[-1]
        # Note the reversed order for the bboxes tran matrices.
        bboxes_tran_mat = tf.scan(matmul, bboxes_tran_mats[::-1])[-1]
        return image_dest_tran_mat, bboxes_tran_mat

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        image_dest_tran_mat, bboxes_tran_mat = self._get_mats(
            image, self._rand_fn
        )
        return super().__call__(
            image, bboxes, labels, image_dest_tran_mat, bboxes_tran_mat,
            self._interpolation, self._fill_value
        )


class TFRandomFlipLeftRight(TFRandomTransform):

    def __init__(
            self,
            probability: float = 0.5,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(
            tf_flip_left_right, probability, seed, "TFRandomFlipLeftRight", True
        )

    def _get_mats(
            self,
            image: tf.Tensor,
            rand_fn: Callable[..., tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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
            probability: float = 0.5,
            seed: Optional[int] = None
    ) -> None:
        super().__init__(
            tf_flip_up_down, probability, seed, "TFRandomFlipUpDown", True
        )

    def _get_mats(
            self,
            image: tf.Tensor,
            rand_fn: Callable[..., tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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
            interpolation: Interpolation = Interpolation.BILINEAR,
            fill_value: float = 0.0,
            probability: float = 0.9,
            seed: Optional[int] = None
    ) -> None:
        _check_input_range(angle_deg_range, None, "angle_deg_range")
        super().__init__(tf_rotate, probability, seed, "TFRandomRotate", True)
        self.angle_deg_range = angle_deg_range
        self.interpolation = interpolation
        self.fill_value = fill_value

    def _get_angle_deg(self, rand_fn: Callable[..., tf.Tensor]) -> tf.Tensor:
        return _tf_convert(self.angle_deg_range[1] - self.angle_deg_range[0]) \
               * rand_fn() + _tf_convert(self.angle_deg_range[0])

    def _get_mats(
            self,
            image: tf.Tensor,
            rand_fn: Callable[..., tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return _get_rotate_mats(
            self._get_angle_deg(rand_fn), _tf_convert, tf.cos, tf.sin
        )

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return super().__call__(
            image, bboxes, labels, self._get_angle_deg(self._rand_fn),
            self.interpolation, self.fill_value
        )


class TFRandomShear(TFRandomTransform):

    def __init__(
            self,
            angle_deg_range: Tuple[float, float] = (-10.0, 10.0),
            interpolation: Interpolation = Interpolation.BILINEAR,
            fill_value: float = 0.0,
            probability: float = 0.9,
            seed: Optional[int] = None
    ) -> None:
        _check_input_range(angle_deg_range, (-90.0, 90.0), "angle_deg_range")
        super().__init__(tf_shear, probability, seed, "TFRandomShear", True)
        self.angle_deg_range = angle_deg_range
        self.interpolation = interpolation
        self.fill_value = fill_value

    def _get_angle_deg(self, rand_fn: Callable[..., tf.Tensor]) -> tf.Tensor:
        return _tf_convert(self.angle_deg_range[1] - self.angle_deg_range[0]) \
               * rand_fn() + _tf_convert(self.angle_deg_range[0])

    def _get_mats(
            self,
            image: tf.Tensor,
            rand_fn: Callable[..., tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return _get_shear_mats(
            self._get_angle_deg(rand_fn), _tf_convert, tf.tan
        )

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return super().__call__(
            image, bboxes, labels, self._get_angle_deg(self._rand_fn),
            self.interpolation, self.fill_value, False
        )


class TFRandomTranslate(TFRandomTransform):

    def __init__(
            self,
            translate_height_fraction_range: Tuple[float, float] = (-0.1, 0.1),
            translate_width_fraction_range: Tuple[float, float] = (-0.1, 0.1),
            interpolation: Interpolation = Interpolation.BILINEAR,
            fill_value: float = 0.0,
            probability: float = 0.9,
            seed: Optional[int] = None
    ) -> None:
        _check_input_range(
            translate_height_fraction_range, (-1.0, 1.0),
            "translate_height_fraction_range"
        )
        _check_input_range(
            translate_width_fraction_range, (-1.0, 1.0),
            "translate_width_fraction_range"
        )
        super().__init__(
            tf_translate, probability, seed, "TFRandomTranslate", True
        )
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range
        self.interpolation = interpolation
        self.fill_value = fill_value

    def _get_translate_height_and_width(
            self,
            image: tf.Tensor,
            rand_fn: Callable[..., tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        height_fraction, width_fraction = _get_random_size_fractions(
            self.translate_height_fraction_range,
            self.translate_width_fraction_range,
            rand_fn, _tf_convert
        )
        translate_height = _tf_convert(tf.shape(image)[0]) * height_fraction
        translate_width = _tf_convert(tf.shape(image)[1]) * width_fraction
        return translate_height, translate_width

    def _get_mats(
            self,
            image: tf.Tensor,
            rand_fn: Callable[..., tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        translate_height, translate_width = \
            self._get_translate_height_and_width(image, rand_fn)
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
            self._get_translate_height_and_width(image, self._rand_fn)
        return super().__call__(
            image, bboxes, labels, translate_height, translate_width,
            self.interpolation, self.fill_value, False
        )


class TFRandomCrop(TFRandomTransform):

    def __init__(
            self,
            crop_height_fraction_range: Tuple[float, float] = (0.8, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.8, 0.9),
            probability: float = 0.9,
            seed: Optional[int] = None
    ) -> None:
        _check_input_range(
            crop_height_fraction_range, (0.0, 1.0), "crop_height_fraction_range"
        )
        _check_input_range(
            crop_width_fraction_range, (0.0, 1.0), "crop_width_fraction_range"
        )
        super().__init__(tf_crop, probability, seed, "TFRandomCrop", False)
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
        offset_height, offset_width, crop_height, crop_width = \
            _tf_get_crop_inputs(
                tf.shape(image)[0], tf.shape(image)[1],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                self._rand_fn
            )

        return super().__call__(
            image, bboxes, labels,
            offset_height, offset_width, crop_height, crop_width, False
        )


class TFResize:

    def __init__(
            self,
            dest_size: Tuple[int, int],
            interpolation: Interpolation = Interpolation.BILINEAR
    ) -> None:
        self.dest_size = dest_size
        self.interpolation = interpolation
        self.name = "TFResize"
        self.is_affine = False

    def __call__(
            self,
            image: tf.Tensor,
            bboxes: tf.Tensor,
            labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return tf_resize(image, bboxes, labels, self.dest_size, self.interpolation)
