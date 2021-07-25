#!/usr/bin/env python3
"""
TF test.
"""

import numpy as np
import tensorflow as tf

import targetran as tt

from scipy import misc


def main() -> None:

    images = np.array([misc.face() for _ in range(3)], dtype=np.float32)
    bboxes_ragged = np.array([
        np.array([
            [214, 223, 10, 11],
            [345, 230, 21, 9],
        ], dtype=np.float32),
        np.array([], dtype=np.float32).reshape(-1, 4),
        np.array([
            [104, 151, 22, 10],
        ], dtype=np.float32),
    ], dtype=object)
    labels_ragged = np.array([
        np.array([[0], [1]], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([[2]], dtype=np.float32),
    ])

    tf_images = tf.convert_to_tensor(images)
    tf_bboxes_ragged = tf.ragged.constant(
        bboxes_ragged.tolist(), inner_shape=(4,)
    )
    tf_labels_ragged = tf.ragged.constant(
        labels_ragged.tolist(), inner_shape=(1,)
    )

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(tf_images),
        tf.data.Dataset.from_tensor_slices(tf_bboxes_ragged),
        tf.data.Dataset.from_tensor_slices(tf_labels_ragged)
    ))

    for sample in ds:
        image, bboxes, labels = sample
        print(f"image shape: {image.get_shape()}")
        print(f"bboxes shape: {bboxes.get_shape()}")
        print(f"labels shape: {labels.get_shape()}")

    print("--------")

    ds = ds.map(tt.TFRandomRotate(probability=1.0))

    for sample in ds:
        image, bboxes, labels = sample
        print(f"transformed image shape: {image.get_shape()}")
        print(f"transformed bboxes shape: {bboxes.get_shape()}")
        print(f"transformed bboxes: {bboxes}")
        print(f"transformed labels shape: {labels.get_shape()}")
        print(f"transformed labels: {labels}")

    print("--------")

    ds = ds.batch(2)

    for batch in ds:
        image_batch, bboxes_batch, labels_batch = batch
        print(f"transformed image batch shape: {image_batch.get_shape()}")
        print(f"transformed bboxes batch shape: {bboxes_batch.get_shape()}")
        print(f"transformed bboxes batch: {bboxes_batch}")
        print(f"transformed labels batch shape: {labels_batch.get_shape()}")
        print(f"transformed labels batch: {labels_batch}")


if __name__ == "__main__":
    main()
