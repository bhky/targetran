#!/usr/bin/env python3
"""
TensorFlow Dataset test.
"""
from typing import Sequence, Tuple

import numpy as np
import numpy.typing

import targetran.tf

NDAnyArray = np.typing.NDArray[np.float_]


def make_np_data() -> Tuple[Sequence[NDAnyArray],
                            Sequence[NDAnyArray],
                            Sequence[NDAnyArray]]:
    image_seq = [np.random.rand(480, 512, 3) for _ in range(3)]

    bboxes_seq = [
        np.array([
            [214, 223, 10, 11],
            [345, 230, 21, 9],
        ]),
        np.array([]),
        np.array([
            [104, 151, 22, 10],
            [99, 132, 20, 15],
            [340, 220, 31, 12],
        ]),
    ]

    labels_seq = [
        np.array([0, 1]),
        np.array([]),
        np.array([2, 3, 0]),
    ]

    return image_seq, bboxes_seq, labels_seq


def main() -> None:
    image_seq, bboxes_seq, labels_seq = make_np_data()

    ds = targetran.tf.to_tf_dataset(image_seq, bboxes_seq, labels_seq)

    print("-------- Raw data --------")

    for example in ds:
        image, bboxes, labels = example
        print(f"image shape: {image.get_shape()}")
        print(f"bboxes shape: {bboxes.get_shape()}")
        print(f"labels shape: {labels.get_shape()}")
        print("=========")

    print("-------- Random transform --------")

    ds = ds \
        .map(targetran.tf.TFRandomRotate(probability=1.0)) \
        .map(targetran.tf.TFRandomShear(probability=1.0)) \
        .map(targetran.tf.TFRandomTranslate(probability=1.0)) \
        .map(targetran.tf.TFRandomFlipUpDown(probability=1.0)) \
        .map(targetran.tf.TFRandomFlipLeftRight(probability=1.0)) \
        .map(targetran.tf.TFRandomCrop(probability=1.0))

    for example in ds:
        image, bboxes, labels = example
        print(f"transformed image shape: {image.get_shape()}")
        print(f"transformed bboxes shape: {bboxes.get_shape()}")
        print(f"transformed bboxes: {bboxes.numpy().tolist()}")
        print(f"transformed labels shape: {labels.get_shape()}")
        print(f"transformed labels: {labels.numpy().tolist()}")
        print("=========")

    print("-------- Random transform with combine-affine --------")

    ds = targetran.tf.to_tf_dataset(image_seq, bboxes_seq, labels_seq)

    affine_transforms = targetran.tf.TFCombineAffine([
        targetran.tf.TFRandomRotate(probability=1.0),
        targetran.tf.TFRandomShear(probability=1.0),
        targetran.tf.TFRandomTranslate(probability=1.0),
        targetran.tf.TFRandomFlipUpDown(probability=1.0),
        targetran.tf.TFRandomFlipLeftRight(probability=1.0),
    ], probability=1.0)

    ds = ds \
        .map(targetran.tf.TFRandomCrop(probability=1.0)) \
        .map(affine_transforms) \
        .map(targetran.tf.TFResize((256, 256)))

    for example in ds:
        image, bboxes, labels = example
        print(f"transformed image shape: {image.get_shape()}")
        print(f"transformed bboxes shape: {bboxes.get_shape()}")
        print(f"transformed bboxes: {bboxes.numpy().tolist()}")
        print(f"transformed labels shape: {labels.get_shape()}")
        print(f"transformed labels: {labels.numpy().tolist()}")
        print("=========")

    print("-------- Batching --------")

    ds = ds.padded_batch(2, padding_values=np.nan)

    for batch in ds:
        image_batch, bboxes_batch, labels_batch = batch
        print(f"transformed image batch shape: {image_batch.get_shape()}")
        print(f"transformed bboxes batch shape: {bboxes_batch.get_shape()}")
        print(f"transformed bboxes batch: {bboxes_batch.numpy().tolist()}")
        print(f"transformed labels batch shape: {labels_batch.get_shape()}")
        print(f"transformed labels batch: {labels_batch.numpy().tolist()}")
        print("=========")


if __name__ == "__main__":
    main()
