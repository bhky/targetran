#!/usr/bin/env python3
"""
Examples.
"""

import numpy as np
import tensorflow as tf

import targetran as tt

from scipy import misc


images = np.array([misc.face() for _ in range(3)])
bboxes_ragged = np.array([
    np.array([
        [14, 23, 10, 11],
        [45, 30, 21, 9],
    ]),
    np.array([
        [38, 31, 12, 12],
    ]),
    np.array([
        [4, 51, 22, 10]
    ]),
], dtype=object)

tf_images = tf.convert_to_tensor(images)
tf_bboxes_ragged = tf.ragged.constant(bboxes_ragged)

ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(images),
    tf.data.Dataset.from_tensor_slices(tf_bboxes_ragged)
))

batch_size = 2
ds = ds\
    .batch(batch_size, drop_remainder=True)\
    .map(tt.TFRandomRotate(batch_size))
