![logo](logo/targetran_logo.png)

# Targetran

- Data augmentation library for object detection or image classification 
  model training. 
- Simple Python API to transform both the images and 
  the target rectangular bounding-boxes.
- Dataset-idiomatic implementation for TensorFlow and PyTorch.

![example](docs/example.png)

(Figure produced by the example code [here](examples/run_tf_dataset_example.py).)

# Usage

For object detection training, which is the primary usage here, 
one will typically have the following data:
- images
- bounding-boxes for each image
- label for each bounding-box

Some dummy data is created for illustration. Please note the required format.
```python
import numpy as np

# Each image could have different sizes, but they must all have 3 channels.
# The channel-last format is used, i.e., (image_height, image_width, num_channels).
image_list = [np.random.rand(480, 512, 3) for _ in range(3)]

# The bounding-boxes (bboxes) are given as a sequence of Numpy arrays (or TF tensors).
# Each array contains the bounding-bboxes for one corresponding image.
#
# Each bbox is represented by: [top_left_x, top_left_y, width, height].
# 
# In case an image has no bboxes, an empty array should be provided.
bboxes_list = [
    np.array([
        [214, 223, 10, 11],  # Image with 2 bboxes.
        [345, 230, 21, 9],
    ], dtype=np.float32),
    np.array([], dtype=np.float32).reshape(-1, 4),  # Empty array for image with no bboxes.
    np.array([
        [104, 151, 22, 10],  # Image with 3 bboxes.
        [99, 132, 20, 15],
        [340, 220, 31, 12],
    ], dtype=np.float32),
]

# Labels for the bboxes are also given as a sequence of Numpy arrays (or TF tensors).
# The number of bboxes and labels should match, and again an empty array indicates no bboxes/labels.
labels_list = [
    np.array([0, 1], dtype=np.float32),  # 2 labels.
    np.array([], dtype=np.float32),  # No labels.
    np.array([2, 3, 0], dtype=np.float32),  # 3 labels.
]

# During operation, all the data values are to be converted float32.
```

With the data ready, the usage of Targetran with TensorFlow and Pytorch 
is presented below.

## TensorFlow

```python
import tensorflow as tf

from targetran.tf import (
    seqs_to_tf_dataset,
    TFCombineAffine,
    TFRandomFlipLeftRight,
    TFRandomFlipUpDown,    
    TFRandomRotate,
    TFRandomShear,
    TFRandomCrop,
    TFRandomTranslate,
    TFResize,
)

# Convert the above data sequences into a TensorFlow Dataset.
ds = seqs_to_tf_dataset(image_list, bboxes_list, labels_list)

# The affine transformations can be combined for better performance.
# Note that cropping and resizing are not affine.
affine_transform = TFCombineAffine([
    TFRandomRotate(),
    TFRandomShear(),
    TFRandomTranslate(),
    TFRandomFlipLeftRight(),
    TFRandomFlipUpDown(),
])

# Typical application.
auto_tune = tf.data.AUTOTUNE
ds = ds \
    .map(TFRandomCrop(), num_parallel_calls=auto_tune) \
    .map(affine_transform, num_parallel_calls=auto_tune) \
    .map(TFResize((256, 256)), num_parallel_calls=auto_tune)

# In the `map` call, parameter `num_parallel_calls` can be set to,
# e.g., tf.data.AUTOTUNE, for better performance. 
# See the docs for TensorFlow Dataset.
```

