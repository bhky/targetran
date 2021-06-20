"""
Unit tests.
"""

import numpy as np
import unittest

from ._transform import _flip_left_right, _flip_up_down, _rotate_90_clockwise


IMAGES = np.array([
    [[[1], [2], [3]],
     [[4], [5], [6]],
     [[7], [8], [9]]],
    [[[11], [12], [13]],
     [[14], [15], [16]],
     [[17], [18], [19]]],
])

BBOXES_LIST = [
    np.array([
        [1, 0, 2, 2],
        [0, 1, 3, 2],
    ]),
    np.array([
        [0, 0, 2, 3],
    ])
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
        ])
        expected_bboxes_list = [
            np.array([
                [0, 0, 2, 2],
                [0, 1, 3, 2],
            ]),
            np.array([
                [1, 0, 2, 3],
            ])
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
        ])
        expected_bboxes_list = [
            np.array([
                [1, 1, 2, 2],
                [0, 0, 3, 2],
            ]),
            np.array([
                [0, 0, 2, 3],
            ])
        ]

        self.assertTrue(
            np.array_equal(expected_images, images)
        )
        for expected_bboxes, bboxes in zip(expected_bboxes_list, bboxes_list):
            self.assertTrue(
                np.array_equal(expected_bboxes, bboxes)
            )

    def test_rotate_90_clockwise(self) -> None:

        images, bboxes_list = _rotate_90_clockwise(
            IMAGES, BBOXES_LIST,
            np.shape, np.transpose, np.concatenate, np.split, np.reshape
        )

        expected_images = np.array([
            [[[7], [4], [1]],
             [[8], [5], [2]],
             [[9], [6], [3]]],
            [[[17], [14], [11]],
             [[18], [15], [12]],
             [[19], [16], [13]]],
        ])
        expected_bboxes_list = [
            np.array([
                [1, 1, 2, 2],
                [0, 0, 2, 3],
            ]),
            np.array([
                [0, 0, 3, 2],
            ])
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
