"""
API for Numpy usage.
"""

from typing import Any, Callable, Tuple

import numpy as np  # type: ignore


from ._functional import (
    _map_single,
    _np_convert,
    _np_ragged_to_list,
    _np_list_to_ragged,
    _np_stack_bboxes,
    _np_resize_image,
    _np_boolean_mask,
    _np_logical_and,
    _np_pad_images,
    _np_make_bboxes_ragged
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


def flip_left_right(
        images: np.ndarray,
        bboxes_ragged: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return _flip_left_right(
        images, bboxes_ragged,
        np.shape, _np_convert, _np_stack_bboxes, np.concatenate,
        _np_make_bboxes_ragged
    )


def flip_up_down(
        images: np.ndarray,
        bboxes_ragged: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return _flip_up_down(
        images, bboxes_ragged,
        np.shape, _np_convert, _np_stack_bboxes, np.concatenate,
        _np_make_bboxes_ragged
    )


def rotate_90(
        images: np.ndarray,
        bboxes_ragged: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return _rotate_90(
        images, bboxes_ragged,
        np.shape, _np_convert, np.transpose, _np_stack_bboxes, np.concatenate,
        _np_make_bboxes_ragged
    )


def resize(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        dest_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    image_list = [image for image in images]
    bboxes_list = _np_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _resize_single, image_list, bboxes_list, None,
        dest_size, np.shape, _np_resize_image, _np_convert, np.concatenate
    )
    images = _np_convert(image_list)
    bboxes_ragged = _np_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


def rotate_90_and_pad(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return _rotate_90_and_pad(
        images, bboxes_ragged,
        np.shape, _np_convert, np.transpose, _np_stack_bboxes, np.concatenate,
        np.where, np.ceil, np.floor, _np_pad_images,
        _np_make_bboxes_ragged
    )


def rotate_90_and_pad_and_resize(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = int(np.shape(images)[1]), int(np.shape(images)[2])
    images, bboxes_ragged = rotate_90_and_pad(images, bboxes_ragged)
    return resize(images, bboxes_ragged, (height, width))


def _np_get_random_crop_inputs(
        image_height: int,
        image_width: int,
        height_fraction_range: Tuple[float, float],
        width_fraction_range: Tuple[float, float],
        rand_fn: Callable[..., np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _get_random_crop_inputs(
        image_height, image_width, height_fraction_range, width_fraction_range,
        rand_fn, _np_convert, np.rint
    )


def crop_and_resize(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        offset_heights: np.ndarray,
        offset_widths: np.ndarray,
        cropped_image_heights: np.ndarray,
        cropped_image_widths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    image_list = [image for image in images]
    bboxes_list = _np_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _crop_single, image_list, bboxes_list,
        [offset_heights, offset_widths,
         cropped_image_heights, cropped_image_widths],
        np.shape, np.reshape, _np_convert, np.concatenate,
        _np_logical_and, np.squeeze, _np_boolean_mask
    )
    image_list, bboxes_list = _map_single(
        _resize_single, image_list, bboxes_list, None,
        np.shape(images)[1:3], np.shape, _np_resize_image,
        _np_convert, np.concatenate
    )
    images = _np_convert(image_list)
    bboxes_ragged = _np_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


def translate(
        images: np.ndarray,
        bboxes_ragged: np.ndarray,
        translate_heights: np.ndarray,
        translate_widths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    image_list = [image for image in images]
    bboxes_list = _np_ragged_to_list(bboxes_ragged)
    image_list, bboxes_list = _map_single(
        _translate_single, image_list, bboxes_list,
        [translate_heights, translate_widths],
        np.shape, np.reshape, _np_convert, np.where, np.abs, np.concatenate,
        _np_logical_and, np.expand_dims, np.squeeze, _np_boolean_mask,
        _np_pad_images
    )
    images = _np_convert(image_list)
    bboxes_ragged = _np_list_to_ragged(bboxes_list)
    return images, bboxes_ragged


class Resize:

    def __init__(self, dest_size: Tuple[int, int]) -> None:
        self.dest_size = dest_size

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return resize(images, bboxes_ragged, self.dest_size)


class RandomTransform:

    def __init__(
            self,
            np_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
            probability: float,
            seed: int,
    ) -> None:
        self._np_fn = np_fn
        self.probability = probability
        self.rng = np.random.default_rng(seed=seed)

    def call(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:

        transformed_images, transformed_bboxes_ragged = self._np_fn(
            images, bboxes_ragged, *args, **kwargs
        )

        rand = self.rng.random(size=np.shape(images)[0])
        is_used = rand < self.probability

        final_images = np.where(is_used, transformed_images, images)
        final_bboxes_ragged_list = [
            transformed_bboxes_ragged[i] if is_used[i] else bboxes_ragged[i]
            for i in range(len(bboxes_ragged))
        ]

        return final_images, np.array(final_bboxes_ragged_list, dtype=object)


class RandomFlipLeftRight(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(flip_left_right, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomFlipUpDown(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(flip_up_down, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomRotate90(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(rotate_90, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomRotate90AndResize(RandomTransform):

    def __init__(self, probability: float = 0.5, seed: int = 0) -> None:
        super().__init__(rotate_90_and_pad_and_resize, probability, seed)

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().call(images, bboxes_ragged)


class RandomCropAndResize(RandomTransform):

    def __init__(
            self,
            crop_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            crop_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(crop_and_resize, probability, seed)
        self.crop_height_fraction_range = crop_height_fraction_range
        self.crop_width_fraction_range = crop_width_fraction_range

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        images_shape = np.shape(images)

        def rand_fn() -> np.ndarray:
            return self.rng.random(images_shape[0])

        offset_heights, offset_widths, cropped_heights, cropped_widths = \
            _np_get_random_crop_inputs(
                images_shape[1], images_shape[2],
                self.crop_height_fraction_range,
                self.crop_width_fraction_range,
                rand_fn
            )

        return super().call(
            images, bboxes_ragged,
            offset_heights, offset_widths, cropped_heights, cropped_widths
        )


class RandomTranslate(RandomTransform):

    def __init__(
            self,
            translate_height_fraction_range: Tuple[float, float] = (0.6, 0.9),
            translate_width_fraction_range: Tuple[float, float] = (0.6, 0.9),
            probability: float = 0.5,
            seed: int = 0
    ) -> None:
        super().__init__(translate, probability, seed)
        self.translate_height_fraction_range = translate_height_fraction_range
        self.translate_width_fraction_range = translate_width_fraction_range

    def __call__(
            self,
            images: np.ndarray,
            bboxes_ragged: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        images_shape = np.shape(images)

        def rand_fn() -> np.ndarray:
            return self.rng.random(images_shape[0])

        height_fractions, width_fractions = _get_random_size_fractions(
            self.translate_height_fraction_range,
            self.translate_width_fraction_range,
            rand_fn, _np_convert
        )

        translate_heights = images_shape[1] * height_fractions
        translate_widths = images_shape[2] * width_fractions

        return super().call(
            images, bboxes_ragged, translate_heights, translate_widths
        )
