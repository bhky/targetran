"""
Unit tests.
"""

import numpy as np
import unittest

from ._transform import _flip_left_right, _flip_up_down, _rotate_90
from ._transform import _crop_and_resize
from ._functional import _np_resize, _np_boolean_mask
from ._functional import _np_multiply, _np_logical_and
from ._functional import _np_array_map, _np_concat_map


ORIGINAL_IMAGES = np.array([
    [[[1], [2], [3]],
     [[4], [5], [6]],
     [[7], [8], [9]]],
    [[[11], [12], [13]],
     [[14], [15], [16]],
     [[17], [18], [19]]],
    [[[21], [22], [23]],
     [[24], [25], [26]],
     [[27], [28], [29]]],
])

ORIGINAL_BBOXES_LIST = [
    np.array([
        [1, 0, 2, 2],
        [0, 1, 3, 2],
    ]),
    np.array([
        [0, 0, 2, 3],
    ]),
    np.array([]).reshape(-1, 4),
]


class TestTransform(unittest.TestCase):

    def test_flip_left_right(self) -> None:

        images, bboxes_list = _flip_left_right(
            ORIGINAL_IMAGES, ORIGINAL_BBOXES_LIST,
            np.shape, np.concatenate, np.split, np.reshape
        )

        expected_images = np.array([
            [[[3], [2], [1]],
             [[6], [5], [4]],
             [[9], [8], [7]]],
            [[[13], [12], [11]],
             [[16], [15], [14]],
             [[19], [18], [17]]],
            [[[23], [22], [21]],
             [[26], [25], [24]],
             [[29], [28], [27]]],
        ])
        expected_bboxes_list = [
            np.array([
                [0, 0, 2, 2],
                [0, 1, 3, 2],
            ]),
            np.array([
                [1, 0, 2, 3],
            ]),
            np.array([]).reshape(-1, 4),
        ]

        self.assertTrue(
            np.array_equal(expected_images, images)
        )
        for expected_bboxes, bboxes in zip(expected_bboxes_list, bboxes_list):
            self.assertTrue(
                np.array_equal(expected_bboxes, bboxes)
            )

    def test_flip_up_down(self) -> None:

        images, bboxes_list = _flip_up_down(
            ORIGINAL_IMAGES, ORIGINAL_BBOXES_LIST,
            np.shape, np.concatenate, np.split, np.reshape
        )

        expected_images = np.array([
            [[[7], [8], [9]],
             [[4], [5], [6]],
             [[1], [2], [3]]],
            [[[17], [18], [19]],
             [[14], [15], [16]],
             [[11], [12], [13]]],
            [[[27], [28], [29]],
             [[24], [25], [26]],
             [[21], [22], [23]]],
        ])
        expected_bboxes_list = [
            np.array([
                [1, 1, 2, 2],
                [0, 0, 3, 2],
            ]),
            np.array([
                [0, 0, 2, 3],
            ]),
            np.array([]).reshape(-1, 4),
        ]

        self.assertTrue(
            np.array_equal(expected_images, images)
        )
        for expected_bboxes, bboxes in zip(expected_bboxes_list, bboxes_list):
            self.assertTrue(
                np.array_equal(expected_bboxes, bboxes)
            )

    def test_rotate_90(self) -> None:

        images, bboxes_list = _rotate_90(
            ORIGINAL_IMAGES, ORIGINAL_BBOXES_LIST,
            np.shape, np.transpose, np.concatenate, np.split, np.reshape
        )

        expected_images = np.array([
            [[[3], [6], [9]],
             [[2], [5], [8]],
             [[1], [4], [7]]],
            [[[13], [16], [19]],
             [[12], [15], [18]],
             [[11], [14], [17]]],
            [[[23], [26], [29]],
             [[22], [25], [28]],
             [[21], [24], [27]]],
        ])
        expected_bboxes_list = [
            np.array([
                [0, 0, 2, 2],
                [1, 0, 2, 3],
            ]),
            np.array([
                [0, 1, 3, 2],
            ]),
            np.array([]).reshape(-1, 4),
        ]

        self.assertTrue(
            np.array_equal(expected_images, images)
        )
        for expected_bboxes, bboxes in zip(expected_bboxes_list, bboxes_list):
            self.assertTrue(
                np.array_equal(expected_bboxes, bboxes)
            )

    def test_crop_and_resize(self) -> None:

        original_images = np.random.rand(4, 128, 128, 3)
        original_bboxes_list = [
            np.array([
                [64, 52, 20, 24],
                [44, 48, 12, 8],
            ]),
            np.array([
                [24, 12, 20, 24],
                [108, 120, 12, 8],
            ]),
            np.array([
                [108, 120, 12, 8],
            ]),
            np.array([]).reshape(-1, 4),
        ]

        x_offset_fractions = np.array([0.25, -0.25, -0.25, 0.25])
        y_offset_fractions = np.array([0.25, -0.25, -0.25, 0.25])
        f = 4 / 3
        expected_bboxes_list = [
            np.array([
                [32 * f, 20 * f, 20 * f, 24 * f],
                [12 * f, 16 * f, 12 * f, 8 * f],
            ]),
            np.array([
                [56 * f, 44 * f, 20 * f, 24 * f],
            ]),
            np.array([]).reshape(-1, 4),
            np.array([]).reshape(-1, 4),
        ]

        _, bboxes_list = _crop_and_resize(
            original_images, original_bboxes_list,
            x_offset_fractions, y_offset_fractions,
            np.shape, _np_multiply, np.rint, np.abs, np.where, np.asarray,
            _np_array_map, _np_resize, _np_concat_map,
            np.concatenate, _np_logical_and, np.squeeze, _np_boolean_mask,
            np.split, np.reshape
        )

        for expected_bboxes, bboxes in zip(expected_bboxes_list, bboxes_list):
            self.assertTrue(
                np.allclose(expected_bboxes, bboxes)
            )


if __name__ == "__main__":
    unittest.main()
