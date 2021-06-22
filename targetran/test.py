"""
Unit tests.
"""

import numpy as np
import unittest

from ._transform import _flip_left_right, _flip_up_down, _rotate_90


IMAGES = np.array([
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

BBOXES_LIST = [
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
            IMAGES, BBOXES_LIST,
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
            IMAGES, BBOXES_LIST,
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
            IMAGES, BBOXES_LIST,
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


if __name__ == "__main__":
    unittest.main()
