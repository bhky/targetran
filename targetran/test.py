"""
Unit tests.
"""

from typing import Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import unittest

from ._transform import (
    _np_flip_left_right,
    _np_flip_up_down,
    _np_rotate_90,
    _np_rotate_90_and_pad,
    _np_crop_and_resize,
    _tf_flip_left_right,
    _tf_flip_up_down,
    _tf_rotate_90,
    _tf_rotate_90_and_pad,
    _tf_crop_and_resize
)


ORIGINAL_IMAGES = np.array([
    [[[1], [2], [3]],
     [[4], [5], [6]],
     [[7], [8], [9]],
     [[10], [11], [12]]],
    [[[11], [12], [13]],
     [[14], [15], [16]],
     [[17], [18], [19]],
     [[20], [21], [22]]],
    [[[21], [22], [23]],
     [[24], [25], [26]],
     [[27], [28], [29]],
     [[30], [31], [32]]],
], dtype=np.float32)

ORIGINAL_BBOXES_RAGGED = np.array([
    np.array([
        [1, 0, 2, 2],
        [0, 1, 3, 2],
    ], dtype=np.float32),
    np.array([
        [0, 0, 2, 3],
    ], dtype=np.float32),
    np.array([], dtype=np.float32).reshape(-1, 4),
], dtype=object)


def _np_to_tf(
        images: np.ndarray,
        bboxes_ragged: np.ndarray
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Convert Numpy arrays to TF tensors.
    """
    return (
        tf.convert_to_tensor(images, dtype=tf.float32),
        tf.ragged.constant([bboxes.tolist() for bboxes in bboxes_ragged])
    )


TF_ORIGINAL_IMAGES, TF_ORIGINAL_BBOXES_RAGGED = _np_to_tf(
    ORIGINAL_IMAGES, ORIGINAL_BBOXES_RAGGED
)


class TestTransform(unittest.TestCase):

    def test_flip_left_right(self) -> None:

        expected_images = np.array([
            [[[3], [2], [1]],
             [[6], [5], [4]],
             [[9], [8], [7]],
             [[12], [11], [10]]],
            [[[13], [12], [11]],
             [[16], [15], [14]],
             [[19], [18], [17]],
             [[22], [21], [20]]],
            [[[23], [22], [21]],
             [[26], [25], [24]],
             [[29], [28], [27]],
             [[32], [31], [30]]],
        ], dtype=np.float32)
        expected_bboxes_ragged = np.array([
            np.array([
                [0, 0, 2, 2],
                [0, 1, 3, 2],
            ], dtype=np.float32),
            np.array([
                [1, 0, 2, 3],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ], dtype=object)

        # Numpy.
        images, bboxes_ragged = _np_flip_left_right(
            ORIGINAL_IMAGES, ORIGINAL_BBOXES_RAGGED
        )
        self.assertTrue(
            np.array_equal(expected_images, images)
        )
        for expected_bboxes, bboxes in zip(expected_bboxes_ragged,
                                           bboxes_ragged):
            self.assertTrue(np.array_equal(expected_bboxes, bboxes))

        # TF.
        tf_expected_images, tf_expected_bboxes_ragged = _np_to_tf(
            expected_images, expected_bboxes_ragged
        )
        tf_images, tf_bboxes_ragged = _tf_flip_left_right(
            TF_ORIGINAL_IMAGES, TF_ORIGINAL_BBOXES_RAGGED
        )

        self.assertTrue(
            np.array_equal(tf_expected_images.numpy(),
                           tf_images.numpy())
        )
        self.assertTrue(
            np.array_equal(tf_expected_bboxes_ragged.to_list(),
                           tf_bboxes_ragged.to_list())
        )

    def test_flip_up_down(self) -> None:

        expected_images = np.array([
            [[[10], [11], [12]],
             [[7], [8], [9]],
             [[4], [5], [6]],
             [[1], [2], [3]]],
            [[[20], [21], [22]],
             [[17], [18], [19]],
             [[14], [15], [16]],
             [[11], [12], [13]]],
            [[[30], [31], [32]],
             [[27], [28], [29]],
             [[24], [25], [26]],
             [[21], [22], [23]]],
        ], dtype=np.float32)
        expected_bboxes_ragged = np.array([
            np.array([
                [1, 2, 2, 2],
                [0, 1, 3, 2],
            ], dtype=np.float32),
            np.array([
                [0, 1, 2, 3],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ], dtype=object)

        # Numpy.
        images, bboxes_ragged = _np_flip_up_down(
            ORIGINAL_IMAGES, ORIGINAL_BBOXES_RAGGED
        )

        self.assertTrue(np.array_equal(expected_images, images))
        for expected_bboxes, bboxes in zip(expected_bboxes_ragged,
                                           bboxes_ragged):
            self.assertTrue(np.array_equal(expected_bboxes, bboxes))

        # TF.
        tf_expected_images, tf_expected_bboxes_ragged = _np_to_tf(
            expected_images, expected_bboxes_ragged
        )
        tf_images, tf_bboxes_ragged = _tf_flip_up_down(
            TF_ORIGINAL_IMAGES, TF_ORIGINAL_BBOXES_RAGGED
        )

        self.assertTrue(
            np.array_equal(tf_expected_images.numpy(),
                           tf_images.numpy())
        )
        self.assertTrue(
            np.array_equal(tf_expected_bboxes_ragged.to_list(),
                           tf_bboxes_ragged.to_list())
        )

    def test_rotate_90(self) -> None:

        expected_images = np.array([
            [[[3], [6], [9], [12]],
             [[2], [5], [8], [11]],
             [[1], [4], [7], [10]]],
            [[[13], [16], [19], [22]],
             [[12], [15], [18], [21]],
             [[11], [14], [17], [20]]],
            [[[23], [26], [29], [32]],
             [[22], [25], [28], [31]],
             [[21], [24], [27], [30]]],
        ], dtype=np.float32)
        expected_bboxes_ragged = np.array([
            np.array([
                [0, 0, 2, 2],
                [1, 0, 2, 3],
            ], dtype=np.float32),
            np.array([
                [0, 1, 3, 2],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ], dtype=object)

        # Numpy.
        images, bboxes_ragged = _np_rotate_90(
            ORIGINAL_IMAGES, ORIGINAL_BBOXES_RAGGED
        )
        self.assertTrue(
            np.array_equal(expected_images, images)
        )
        for expected_bboxes, bboxes in zip(expected_bboxes_ragged,
                                           bboxes_ragged):
            self.assertTrue(np.array_equal(expected_bboxes, bboxes))

        # TF.
        tf_expected_images, tf_expected_bboxes_ragged = _np_to_tf(
            expected_images, expected_bboxes_ragged
        )
        tf_images, tf_bboxes_ragged = _tf_rotate_90(
            TF_ORIGINAL_IMAGES, TF_ORIGINAL_BBOXES_RAGGED
        )

        self.assertTrue(
            np.array_equal(tf_expected_images.numpy(),
                           tf_images.numpy())
        )
        self.assertTrue(
            np.array_equal(tf_expected_bboxes_ragged.to_list(),
                           tf_bboxes_ragged.to_list())
        )

    def test_rotate_90_and_pad(self) -> None:

        expected_images = np.array([
            [[[0], [0], [0], [0]],
             [[0], [0], [0], [0]],
             [[3], [6], [9], [12]],
             [[2], [5], [8], [11]],
             [[1], [4], [7], [10]],
             [[0], [0], [0], [0]]],
            [[[0], [0], [0], [0]],
             [[0], [0], [0], [0]],
             [[13], [16], [19], [22]],
             [[12], [15], [18], [21]],
             [[11], [14], [17], [20]],
             [[0], [0], [0], [0]]],
            [[[0], [0], [0], [0]],
             [[0], [0], [0], [0]],
             [[23], [26], [29], [32]],
             [[22], [25], [28], [31]],
             [[21], [24], [27], [30]],
             [[0], [0], [0], [0]]],
        ], dtype=np.float32)
        expected_bboxes_ragged = np.array([
            np.array([
                [0, 2, 2, 2],
                [1, 2, 2, 3],
            ], dtype=np.float32),
            np.array([
                [0, 3, 3, 2],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ], dtype=object)

        # Numpy.
        images, bboxes_ragged = _np_rotate_90_and_pad(
            ORIGINAL_IMAGES, ORIGINAL_BBOXES_RAGGED
        )
        self.assertTrue(
            np.array_equal(expected_images, images)
        )
        for expected_bboxes, bboxes in zip(expected_bboxes_ragged,
                                           bboxes_ragged):
            self.assertTrue(np.array_equal(expected_bboxes, bboxes))

        # TF.
        tf_expected_images, tf_expected_bboxes_ragged = _np_to_tf(
            expected_images, expected_bboxes_ragged
        )
        tf_images, tf_bboxes_ragged = _tf_rotate_90_and_pad(
            TF_ORIGINAL_IMAGES, TF_ORIGINAL_BBOXES_RAGGED
        )

        self.assertTrue(
            np.array_equal(tf_expected_images.numpy(),
                           tf_images.numpy())
        )
        self.assertTrue(
            np.array_equal(tf_expected_bboxes_ragged.to_list(),
                           tf_bboxes_ragged.to_list())
        )

    def test_crop_and_resize(self) -> None:

        dummy_images = np.random.rand(4, 128, 128, 3)
        original_bboxes_ragged = np.array([
            np.array([
                [64, 52, 20, 24],
                [44, 48, 12, 8],
            ], dtype=np.float32),
            np.array([
                [24, 12, 20, 24],
                [108, 120, 12, 8],
            ], dtype=np.float32),
            np.array([
                [108, 120, 12, 8],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ], dtype=object)

        x_offset_fractions = np.array([0.25, -0.25, -0.25, 0.2],
                                      dtype=np.float32)
        y_offset_fractions = np.array([0.25, -0.25, -0.25, -0.1],
                                      dtype=np.float32)
        f = 4 / 3
        expected_bboxes_ragged = np.array([
            np.array([
                [32 * f, 20 * f, 20 * f, 24 * f],
                [12 * f, 16 * f, 12 * f, 8 * f],
            ], dtype=np.float32),
            np.array([
                [56 * f, 44 * f, 20 * f, 24 * f],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ], dtype=object)

        # Numpy.
        _, bboxes_ragged = _np_crop_and_resize(
            dummy_images, original_bboxes_ragged,
            x_offset_fractions, y_offset_fractions
        )
        for expected_bboxes, bboxes in zip(expected_bboxes_ragged,
                                           bboxes_ragged):
            self.assertTrue(np.allclose(expected_bboxes, bboxes))

        # TF.
        tf_dummy_images, tf_original_bboxes_ragged = _np_to_tf(
            dummy_images, original_bboxes_ragged
        )
        _, tf_expected_bboxes_ragged = _np_to_tf(
            dummy_images, expected_bboxes_ragged
        )
        _, tf_bboxes_ragged = _tf_crop_and_resize(
            tf_dummy_images, tf_original_bboxes_ragged,
            tf.convert_to_tensor(x_offset_fractions),
            tf.convert_to_tensor(y_offset_fractions)
        )
        for expected_bboxes, bboxes in zip(tf_expected_bboxes_ragged,
                                           tf_bboxes_ragged):
            self.assertTrue(
                np.allclose(expected_bboxes.numpy(), bboxes.numpy())
            )


if __name__ == "__main__":
    unittest.main()
