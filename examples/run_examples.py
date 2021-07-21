#!/usr/bin/env python3
"""
Examples.
"""

import numpy as np
import tensorflow as tf

import targetran as tt

from scipy import misc


images = np.array([misc.face() for _ in range(3)], dtype=np.float32)
bboxes_ragged = np.array([
    np.array([
        [14, 23, 10, 11],
        [45, 30, 21, 9],
    ], dtype=np.float32),
    np.array([], dtype=np.float32).reshape(-1, 4),
    np.array([
        [4, 51, 22, 10],
    ], dtype=np.float32),
], dtype=object)

tf_images = tf.convert_to_tensor(images)
tf_bboxes_ragged = tf.ragged.constant(bboxes_ragged.tolist(), inner_shape=(4,))

ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(images),
    tf.data.Dataset.from_tensor_slices(tf_bboxes_ragged)
))

batch_size = 2
ds = ds.batch(batch_size, drop_remainder=True)\

for d in ds:
    i, b = d
    print(f"image batch shape: {i.get_shape()}")
    print(f"bboxes-ragged batch shape: {b.get_shape()}")

print("--------")

ds = ds.map(tt.TFRandomRotate(batch_size))

print(list(ds.as_numpy_iterator()))
