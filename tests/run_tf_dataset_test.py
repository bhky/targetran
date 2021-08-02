#!/usr/bin/env python3
"""
TensorFlow Dataset test.
"""

from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf

import targetran.tf


def make_np_data() -> Tuple[Sequence[np.ndarray],
                            Sequence[np.ndarray],
                            Sequence[np.ndarray]]:

    image_list = [np.random.rand(480, 512, 3) for _ in range(3)]

    bboxes_list = [
        np.array([
            [214, 223, 10, 11],
            [345, 230, 21, 9],
        ], dtype=np.float32),
        np.array([], dtype=np.float32).reshape(-1, 4),
        np.array([
            [104, 151, 22, 10],
            [99, 132, 20, 15],
            [340, 220, 31, 12],
        ], dtype=np.float32),
    ]

    labels_list = [
        np.array([0, 1], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([2, 3, 0], dtype=np.float32),
    ]

    return image_list, bboxes_list, labels_list


def main() -> None:

    image_list, bboxes_list, labels_list = make_np_data()

    tf_image_list, tf_bboxes_list, tf_labels_list = targetran.tf.np_to_tf(
        image_list, bboxes_list, labels_list
    )

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_image_list)),
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_bboxes_list)),
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_labels_list))
    ))

    print("-------- Raw data --------")

    for sample in ds:
        image, bboxes, labels = sample
        print(f"image shape: {image.get_shape()}")
        print(f"bboxes shape: {bboxes.get_shape()}")
        print(f"labels shape: {labels.get_shape()}")

    print("-------- Random transform --------")

    ds = ds.map(lambda i, b, l: (i.to_tensor(), b.to_tensor(), l))
    ds = ds \
        .map(targetran.tf.TFRandomRotate(probability=1.0)) \
        .map(targetran.tf.TFRandomFlipUpDown(probability=1.0))

    for sample in ds:
        image, bboxes, labels = sample
        print(f"transformed image shape: {image.get_shape()}")
        print(f"transformed bboxes shape: {bboxes.get_shape()}")
        print(f"transformed bboxes: {bboxes.numpy().tolist()}")
        print(f"transformed labels shape: {labels.get_shape()}")
        print(f"transformed labels: {labels.numpy().tolist()}")

    print("-------- Batching --------")

    ds = ds.padded_batch(2, padding_values=np.nan)

    for batch in ds:
        image_batch, bboxes_batch, labels_batch = batch
        print(f"transformed image batch shape: {image_batch.get_shape()}")
        print(f"transformed bboxes batch shape: {bboxes_batch.get_shape()}")
        print(f"transformed bboxes batch: {bboxes_batch.numpy().tolist()}")
        print(f"transformed labels batch shape: {labels_batch.get_shape()}")
        print(f"transformed labels batch: {labels_batch.numpy().tolist()}")


if __name__ == "__main__":
    main()
