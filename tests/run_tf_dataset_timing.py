#!/usr/bin/env python3
"""
TensorFlow Dataset timing.
"""

from typing import List
from timeit import default_timer as timer

import tensorflow as tf

from targetran.tf import (
    seqs_to_tf_dataset,
    TFRandomFlipLeftRight,
    TFRandomRotate,
    TFRandomShear,
    TFRandomCrop,
    TFRandomTranslate,
    TFResize
)

AUTO = tf.data.AUTOTUNE
rng = tf.random.Generator.from_seed(42)


def make_ds() -> tf.data.Dataset:
    sample_size = 100000
    image_list: List[tf.Tensor] = []
    bboxes_list: List[tf.Tensor] = []
    labels_list: List[tf.Tensor] = []
    print(f"Preparing {sample_size} random samples...")
    for _ in range(sample_size):
        height = rng.uniform(
            shape=(), minval=512, maxval=1024, dtype=tf.int32
        )
        width = rng.uniform(
            shape=(), minval=512, maxval=1024, dtype=tf.int32
        )
        image = rng.uniform(
            shape=(height, width, 3), minval=0, maxval=255, dtype=tf.int32
        )

        num_bboxes = rng.uniform(shape=(), minval=0, maxval=10, dtype=tf.int32)
        bboxes = rng.uniform(
            shape=(num_bboxes, 4), minval=16, maxval=256, dtype=tf.int32
        )
        labels = rng.uniform(
            shape=(num_bboxes,), minval=0, maxval=20, dtype=tf.int32
        )

        image_list.append(image)
        bboxes_list.append(bboxes)
        labels_list.append(labels)
    print("Done preparing random samples.")
    return seqs_to_tf_dataset(image_list, bboxes_list, labels_list)


def main() -> None:
    ds = make_ds()
    ds = ds \
        .map(TFRandomFlipLeftRight(), num_parallel_calls=AUTO) \
        .map(TFRandomRotate(), num_parallel_calls=AUTO) \
        .map(TFRandomShear(), num_parallel_calls=AUTO) \
        .map(TFRandomCrop(), num_parallel_calls=AUTO) \
        .map(TFRandomTranslate(), num_parallel_calls=AUTO) \
        .map(TFResize(dest_size=(256, 256)), num_parallel_calls=AUTO)

    start = timer()
    count = 0
    print("Start...")
    for _ in ds:
        count += 1
        if count == 1000:
            print(f"- Runtime for {count} samples: {timer() - start} s")
    print("--------------")
    print(f"Total runtime for {count} samples: {timer() - start}")


if __name__ == "__main__":
    main()
