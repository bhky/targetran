"""
Unit tests.
"""

import unittest

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

from .np import (
    flip_left_right,
    flip_up_down,
    rotate,
    shear,
    translate,
    crop,
)
from .tf import (
    to_tf,
    seqs_to_tf_dataset,
    tf_flip_left_right,
    tf_flip_up_down,
    tf_rotate,
    tf_shear,
    tf_translate,
    tf_crop,
)

ORIGINAL_IMAGE_SEQ = [
    np.array([
        [[1], [2], [3]],
        [[4], [5], [6]],
        [[7], [8], [9]],
        [[10], [11], [12]]
    ]),
    np.array([
        [[11], [12], [13]],
        [[14], [15], [16]],
        [[17], [18], [19]],
        [[20], [21], [22]]
    ]),
    np.array([
        [[21], [22], [23]],
        [[24], [25], [26]],
        [[27], [28], [29]],
        [[30], [31], [32]]
    ]),
]

ORIGINAL_BBOXES_SEQ = [
    np.array([
        [1, 0, 2, 2],
        [0, 1, 3, 2],
    ]),
    np.array([
        [0, 0, 2, 3],
    ]),
    np.array([]),
]

ORIGINAL_LABELS_SEQ = [
    np.array([0, 1]),
    np.array([2]),
    np.array([]),
]

(
    TF_ORIGINAL_IMAGE_SEQ, TF_ORIGINAL_BBOXES_SEQ, TF_ORIGINAL_LABELS_SEQ
) = to_tf(
    ORIGINAL_IMAGE_SEQ, ORIGINAL_BBOXES_SEQ, ORIGINAL_LABELS_SEQ
)


class TestTransform(unittest.TestCase):

    def test_flip_left_right(self) -> None:

        expected_image_seq = [
            np.array([
                [[3], [2], [1]],
                [[6], [5], [4]],
                [[9], [8], [7]],
                [[12], [11], [10]]
            ], dtype=np.float32),
            np.array([
                [[13], [12], [11]],
                [[16], [15], [14]],
                [[19], [18], [17]],
                [[22], [21], [20]]
            ], dtype=np.float32),
            np.array([
                [[23], [22], [21]],
                [[26], [25], [24]],
                [[29], [28], [27]],
                [[32], [31], [30]]
            ], dtype=np.float32),
        ]
        expected_bboxes_seq = [
            np.array([
                [0, 0, 2, 2],
                [0, 1, 3, 2],
            ], dtype=np.float32),
            np.array([
                [1, 0, 2, 3],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ]
        expected_labels_seq = ORIGINAL_LABELS_SEQ

        # NumPy.
        for i in range(len(ORIGINAL_IMAGE_SEQ)):
            image, bboxes, labels = flip_left_right(
                ORIGINAL_IMAGE_SEQ[i],
                ORIGINAL_BBOXES_SEQ[i],
                ORIGINAL_LABELS_SEQ[i]
            )
            self.assertTrue(
                np.array_equal(expected_image_seq[i], image)
            )
            self.assertTrue(
                np.array_equal(expected_bboxes_seq[i], bboxes)
            )
            self.assertTrue(
                np.array_equal(expected_labels_seq[i], labels)
            )

        # TF.
        (
            tf_expected_image_seq,
            tf_expected_bboxes_seq,
            tf_expected_labels_seq
        ) = to_tf(
            expected_image_seq, expected_bboxes_seq, expected_labels_seq
        )
        for i in range(len(TF_ORIGINAL_LABELS_SEQ)):
            tf_image, tf_bboxes, tf_labels = tf_flip_left_right(
                TF_ORIGINAL_IMAGE_SEQ[i],
                TF_ORIGINAL_BBOXES_SEQ[i],
                TF_ORIGINAL_LABELS_SEQ[i]
            )
            self.assertTrue(
                np.array_equal(tf_expected_image_seq[i].numpy(),
                               tf_image.numpy())
            )
            self.assertTrue(
                np.array_equal(tf_expected_bboxes_seq[i].numpy(),
                               tf_bboxes.numpy())
            )
            self.assertTrue(
                np.array_equal(tf_expected_labels_seq[i].numpy(),
                               tf_labels.numpy())
            )

    def test_flip_up_down(self) -> None:

        # Noted the shifted pixel.
        expected_image_seq = [
            np.array([
                [[7], [8], [9]],
                [[4], [5], [6]],
                [[1], [2], [3]],
                [[0], [0], [0]]
            ], dtype=np.float32),
            np.array([
                [[17], [18], [19]],
                [[14], [15], [16]],
                [[11], [12], [13]],
                [[0], [0], [0]]
            ], dtype=np.float32),
            np.array([
                [[27], [28], [29]],
                [[24], [25], [26]],
                [[21], [22], [23]],
                [[0], [0], [0]]
            ], dtype=np.float32),
        ]
        expected_bboxes_seq = [
            np.array([
                [1, 1, 2, 2],
                [0, 0, 3, 2],
            ], dtype=np.float32),
            np.array([
                [0, 0, 2, 3],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ]
        expected_labels_seq = ORIGINAL_LABELS_SEQ

        # NumPy.
        for i in range(len(ORIGINAL_IMAGE_SEQ)):
            image, bboxes, labels = flip_up_down(
                ORIGINAL_IMAGE_SEQ[i],
                ORIGINAL_BBOXES_SEQ[i],
                ORIGINAL_LABELS_SEQ[i]
            )
            self.assertTrue(
                np.array_equal(expected_image_seq[i], image)
            )
            self.assertTrue(
                np.array_equal(expected_bboxes_seq[i], bboxes)
            )
            self.assertTrue(
                np.array_equal(expected_labels_seq[i], labels)
            )

        # TF.
        (
            tf_expected_image_seq,
            tf_expected_bboxes_seq,
            tf_expected_labels_seq
        ) = to_tf(
            expected_image_seq, expected_bboxes_seq, expected_labels_seq
        )
        for i in range(len(TF_ORIGINAL_LABELS_SEQ)):
            tf_image, tf_bboxes, tf_labels = tf_flip_up_down(
                TF_ORIGINAL_IMAGE_SEQ[i],
                TF_ORIGINAL_BBOXES_SEQ[i],
                TF_ORIGINAL_LABELS_SEQ[i]
            )
            self.assertTrue(
                np.array_equal(tf_expected_image_seq[i].numpy(),
                               tf_image.numpy())
            )
            self.assertTrue(
                np.array_equal(tf_expected_bboxes_seq[i].numpy(),
                               tf_bboxes.numpy())
            )
            self.assertTrue(
                np.array_equal(tf_expected_labels_seq[i].numpy(),
                               tf_labels.numpy())
            )

    def test_rotate(self) -> None:

        original_image_seq = [
            np.array([
                [[1], [2], [3], [4]],
                [[5], [6], [7], [8]],
                [[9], [10], [11], [12]]
            ], dtype=np.float32),
            np.array([
                [[13], [14], [15], [16]],
                [[17], [18], [19], [20]],
                [[21], [22], [23], [24]]
            ], dtype=np.float32),
        ]
        original_bboxes_seq = [
            np.array([
                [1, 0, 2, 2],
                [0, 1, 3, 2],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ]
        original_labels_seq = [
            np.array([1, 2], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

        angles_deg = [90.0, 180.0]

        # Noted the shifted pixel.
        expected_image_seq = [
            np.array([
                [[3], [7], [11], [0]],
                [[2], [6], [10], [0]],
                [[1], [5], [9], [0]]
            ], dtype=np.float32),
            np.array([
                [[23], [22], [21], [0]],
                [[19], [18], [17], [0]],
                [[15], [14], [13], [0]]
            ], dtype=np.float32),
        ]
        expected_bboxes_seq = [
            np.array([
                [0, 0, 2, 2],
                [1, 0, 2, 3],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ]
        expected_labels_seq = original_labels_seq

        # NumPy.
        for i in range(len(original_image_seq)):
            image, bboxes, labels = rotate(
                original_image_seq[i],
                original_bboxes_seq[i],
                original_labels_seq[i],
                angles_deg[i]
            )
            self.assertTrue(
                np.array_equal(expected_image_seq[i], image)
            )
            self.assertTrue(
                np.array_equal(expected_bboxes_seq[i], bboxes)
            )
            self.assertTrue(
                np.array_equal(expected_labels_seq[i], labels)
            )

        # TF.
        (
            tf_original_image_seq,
            tf_original_bboxes_seq,
            tf_original_labels_seq
        ) = to_tf(
            original_image_seq, original_bboxes_seq, original_labels_seq
        )
        (
            tf_expected_image_seq,
            tf_expected_bboxes_seq,
            tf_expected_labels_seq
        ) = to_tf(
            expected_image_seq, expected_bboxes_seq, expected_labels_seq
        )
        for i in range(len(tf_original_image_seq)):
            tf_image, tf_bboxes, tf_labels = tf_rotate(
                tf_original_image_seq[i],
                tf_original_bboxes_seq[i],
                tf_original_labels_seq[i],
                angles_deg[i]
            )
            self.assertTrue(
                np.array_equal(tf_expected_image_seq[i].numpy(),
                               tf_image.numpy())
            )
            self.assertTrue(
                np.array_equal(tf_expected_bboxes_seq[i].numpy(),
                               tf_bboxes.numpy())
            )
            self.assertTrue(
                np.array_equal(tf_expected_labels_seq[i].numpy(),
                               tf_labels.numpy())
            )

    def test_shear(self) -> None:

        dummy_image = np.random.rand(32, 32, 3)
        original_bboxes = np.array([
            [15, 15, 2, 2],
            [15, 0, 2, 2],
        ], dtype=np.float32)
        original_labels = np.array([[0], [1]], dtype=np.float32)

        angle_deg = 45.0

        expected_bboxes = np.array([
            [15, 15, 3, 2],
            [0, 0, 3, 2],
        ], dtype=np.float32)
        expected_labels = original_labels

        # NumPy.
        _, bboxes, labels = shear(
            dummy_image, original_bboxes, original_labels,
            angle_deg
        )
        self.assertTrue(np.array_equal(expected_bboxes, bboxes))
        self.assertTrue(np.array_equal(expected_labels, labels))

        # TF.
        tf_expected_bboxes = tf.convert_to_tensor(
            expected_bboxes, dtype=tf.float32
        )
        tf_expected_labels = tf.convert_to_tensor(
            expected_labels, dtype=tf.float32
        )

        _, tf_bboxes, tf_labels = tf_shear(
            tf.convert_to_tensor(dummy_image, dtype=tf.float32),
            tf.convert_to_tensor(original_bboxes, dtype=tf.float32),
            tf.convert_to_tensor(original_labels, dtype=tf.float32),
            angle_deg
        )
        self.assertTrue(
            np.allclose(tf_expected_bboxes.numpy(), tf_bboxes.numpy())
        )
        self.assertTrue(
            np.allclose(tf_expected_labels.numpy(), tf_labels.numpy())
        )

    def test_translate(self) -> None:

        translate_heights = [-1, 0, 1]
        translate_widths = [0, 1, 1]

        expected_image_seq = [
            np.array([
                [[4], [5], [6]],
                [[7], [8], [9]],
                [[10], [11], [12]],
                [[0], [0], [0]]
            ], dtype=np.float32),
            np.array([
                [[0], [11], [12]],
                [[0], [14], [15]],
                [[0], [17], [18]],
                [[0], [20], [21]]
            ], dtype=np.float32),
            np.array([
                [[0], [0], [0]],
                [[0], [21], [22]],
                [[0], [24], [25]],
                [[0], [27], [28]]
            ], dtype=np.float32),
        ]
        expected_bboxes_seq = [
            np.array([
                [1, 0, 2, 1],
                [0, 0, 3, 2],
            ], dtype=np.float32),
            np.array([
                [1, 0, 2, 3],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ]
        expected_labels_seq = [
            np.array([0, 1], dtype=np.float32),
            np.array([2], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

        # NumPy.
        for i in range(len(ORIGINAL_IMAGE_SEQ)):
            image, bboxes, labels = translate(
                ORIGINAL_IMAGE_SEQ[i],
                ORIGINAL_BBOXES_SEQ[i],
                ORIGINAL_LABELS_SEQ[i],
                translate_heights[i], translate_widths[i]
            )
            self.assertTrue(np.array_equal(expected_image_seq[i], image))
            self.assertTrue(np.array_equal(expected_bboxes_seq[i], bboxes))
            self.assertTrue(np.array_equal(expected_labels_seq[i], labels))

        # TF.
        (
            tf_expected_image_seq,
            tf_expected_bboxes_seq,
            tf_expected_labels_seq
        ) = to_tf(
            expected_image_seq, expected_bboxes_seq, expected_labels_seq
        )

        for i in range(len(TF_ORIGINAL_LABELS_SEQ)):
            tf_image, tf_bboxes, tf_labels = tf_translate(
                TF_ORIGINAL_IMAGE_SEQ[i],
                TF_ORIGINAL_BBOXES_SEQ[i],
                TF_ORIGINAL_LABELS_SEQ[i],
                translate_heights[i], translate_widths[i]
            )
            self.assertTrue(
                np.array_equal(tf_expected_image_seq[i].numpy(),
                               tf_image.numpy())
            )
            self.assertTrue(
                np.array_equal(tf_expected_bboxes_seq[i].numpy(),
                               tf_bboxes.numpy())
            )
            self.assertTrue(
                np.array_equal(tf_expected_labels_seq[i].numpy(),
                               tf_labels.numpy())
            )

    def test_crop(self) -> None:

        dummy_image_seq = [np.random.rand(128, 128, 3) for _ in range(4)]
        original_bboxes_seq = [
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
        ]
        original_labels_seq = [
            np.array([0, 1], dtype=np.float32),
            np.array([2, 1], dtype=np.float32),
            np.array([0], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

        offset_heights = [128 * 0.25, 0.0, 0.0, 99.0]
        offset_widths = [128 * 0.25, 0.0, 0.0, 59.0]
        crop_heights = [128 * 0.75] * 4
        crop_widths = [128 * 0.75] * 4

        expected_bboxes_seq = [
            np.array([
                [32, 20, 20, 24],
                [12, 16, 12, 8],
            ], dtype=np.float32),
            np.array([
                [24, 12, 20, 24],
            ], dtype=np.float32),
            np.array([], dtype=np.float32).reshape(-1, 4),
            np.array([], dtype=np.float32).reshape(-1, 4),
        ]
        expected_labels_seq = [
            np.array([0, 1], dtype=np.float32),
            np.array([2], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

        # NumPy.
        for i in range(len(dummy_image_seq)):
            _, bboxes, labels = crop(
                dummy_image_seq[i],
                original_bboxes_seq[i],
                original_labels_seq[i],
                int(offset_heights[i]), int(offset_widths[i]),
                int(crop_heights[i]), int(crop_widths[i])
            )
            self.assertTrue(
                np.allclose(expected_bboxes_seq[i], bboxes)
            )
            self.assertTrue(
                np.allclose(expected_labels_seq[i], labels)
            )

        # TF.
        (
            tf_dummy_image_seq,
            tf_original_bboxes_seq,
            tf_original_labels_seq
        ) = to_tf(
            dummy_image_seq, original_bboxes_seq, original_labels_seq
        )
        _, tf_expected_bboxes_seq, tf_expected_labels_seq = to_tf(
            dummy_image_seq, expected_bboxes_seq, expected_labels_seq
        )

        for i in range(len(tf_dummy_image_seq)):
            _, tf_bboxes, tf_labels = tf_crop(
                tf_dummy_image_seq[i],
                tf_original_bboxes_seq[i],
                tf_original_labels_seq[i],
                int(offset_heights[i]), int(offset_widths[i]),
                int(crop_heights[i]), int(crop_widths[i])
            )
            self.assertTrue(
                np.allclose(tf_expected_bboxes_seq[i].numpy(),
                            tf_bboxes.numpy())
            )
            self.assertTrue(
                np.allclose(tf_expected_labels_seq[i].numpy(),
                            tf_labels.numpy())
            )


class TestConversion(unittest.TestCase):

    def test_seqs_to_tf_dataset(self) -> None:
        ds = seqs_to_tf_dataset(
            ORIGINAL_IMAGE_SEQ, ORIGINAL_BBOXES_SEQ, ORIGINAL_LABELS_SEQ
        )
        for i, (image, bboxes, labels) in enumerate(ds):
            self.assertTrue(
                np.allclose(ORIGINAL_IMAGE_SEQ[i], image.numpy())
            )
            self.assertTrue(
                np.allclose(ORIGINAL_BBOXES_SEQ[i], bboxes.numpy())
            )
            self.assertTrue(
                np.allclose(ORIGINAL_LABELS_SEQ[i], labels.numpy())
            )

        ds_image_only = seqs_to_tf_dataset(
            ORIGINAL_IMAGE_SEQ, [], []
        )
        for i, (image, _, _) in enumerate(ds_image_only):
            self.assertTrue(
                np.allclose(ORIGINAL_IMAGE_SEQ[i], image.numpy())
            )


if __name__ == "__main__":
    unittest.main()
