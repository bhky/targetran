#!/usr/bin/env python3
"""
TensorFlow Dataset timing.
"""

from typing import Iterator, Tuple

import os
from timeit import default_timer as timer

import tensorflow as tf

from targetran.tf import (
    TFRandomFlipLeftRight,
    TFRandomRotate,
    TFRandomShear,
    TFRandomCrop,
    TFRandomTranslate,
    TFResize
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
AUTO = tf.data.AUTOTUNE
rng = tf.random.Generator.from_seed(42)


def generator() -> Iterator[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """
    Generate random data.
    """
    sample_size = 100000
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
        yield image, bboxes, labels


def main() -> None:
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((None, None, 3)),
            tf.TensorSpec((None, 4)),
            tf.TensorSpec((None,))
        )
    )

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
