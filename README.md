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

## Data format

For object detection model training, which is the primary usage here, the following data are needed.
- `image_list` (Sequence of `np.ndarray` or `tf.Tensor` of shape `(image_height, image_width, num_channels)`):
  - images in channel-last format;
  - image sizes can be different.
- `bboxes_list` (Sequence of `np.ndarray` or `tf.Tensor` of shape `(num_bboxes_per_image, 4)`):
  - each `bboxes` array/tensor represents the bounding-boxes associated with an image;
  - each `bboxes` is given as `[top_left_x, top_left_y, width, height]`;
  - empty array/tensor means no bounding-boxes (and labels) for that image.
- `labels_list` (Sequence of `np.ndarray` or `tf.Tensor` of shape `(num_bboxes_per_image,)`):
  - each `labels` array/tensor represents the bounding-box labels associated with an image;
  - empty array/tensor means no labels (and bounding-boxes) for that image.

Some dummy data are created below for illustration. Please note the required format.
```python
import numpy as np

# Each image could have different sizes, but they must follow the channel-last format, 
# i.e., (image_height, image_width, num_channels).
image_list = [np.random.rand(480, 512, 3) for _ in range(3)]

# The bounding-boxes (bboxes) are given as a sequence of Numpy arrays (or TF tensors).
# Each array contains the bounding-bboxes for one corresponding image.
#
# Each bbox is represented by: [top_left_x, top_left_y, width, height].
# 
# In case an image has no bboxes, an empty array should be provided.
bboxes_list = [
    np.array([  # Image with 2 bboxes.
        [214, 223, 10, 11],
        [345, 230, 21, 9],
    ]),
    np.array([]),  # Empty array for image with no bboxes.
    np.array([  # Image with 3 bboxes.
        [104, 151, 22, 10],
        [99, 132, 20, 15],
        [340, 220, 31, 12],
    ]),
]

# Labels for the bboxes are also given as a sequence of Numpy arrays (or TF tensors).
# The number of bboxes and labels should match. An empty array indicates no bboxes/labels.
labels_list = [
    np.array([0, 1]),  # 2 labels.
    np.array([]),  # No labels.
    np.array([2, 3, 0]),  # 3 labels.
]

# During operation, all the data values will be converted to float32.
```

## Design principles

- Bounding-boxes will always be rectangular with sides parallel to the image frame.
- After transformation, each resulting bounding-box is determined by the smallest 
  rectangle (with sides parallel to the image frame) enclosing the original transformed bounding-box.
- After transformation, resulting bounding-boxes with their centroids outside the 
  image frame will be removed, together with the corresponding labels.

## TensorFlow Dataset usage

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
# Users can have their own way to create the Dataset, as long as for each iteration 
# it returns a tuple of tensors for a single image: (image, bboxes, labels).
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

# In the Dataset `map` call, the parameter `num_parallel_calls` can be set to,
# e.g., tf.data.AUTOTUNE, for better performance. See docs for TensorFlow Dataset.
```

## PyTorch Dataset usage

```python
from typing import Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from targetran.np import (
    CombineAffine,
    RandomFlipLeftRight,
    RandomFlipUpDown,
    RandomRotate,
    RandomShear,
    RandomCrop,
    RandomTranslate,
    Resize,
)
from targetran.utils import Compose


class PTDataset(Dataset):
    """
    A very simple PyTorch Dataset.
    As per common practice, transforms are done on Numpy arrays.
    """
    
    def __init__(
            self,
            image_list: Sequence[np.ndarray],
            bboxes_list: Sequence[np.ndarray],
            labels_list: Sequence[np.ndarray],
            transforms: Optional[Compose]
    ) -> None:
        self.image_list = image_list
        self.bboxes_list = bboxes_list
        self.labels_list = labels_list
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.transforms:
            return self.transforms(
                self.image_list[idx],
                self.bboxes_list[idx],
                self.labels_list[idx]
            )
        return (
            self.image_list[idx],
            self.bboxes_list[idx],
            self.labels_list[idx]
        )


# The affine transformations can be combined for better performance.
# Note that cropping and resizing are not affine.
affine_transform = CombineAffine([
    RandomRotate(),
    RandomShear(),
    RandomTranslate(),
    RandomFlipLeftRight(),
    RandomFlipUpDown(),
])

# The `Compose` here is similar to that from the torchvision package, except 
# that here it also supports callables with multiple inputs and outputs needed
# for objection detection tasks, i.e., (image, bboxes, labels).
transforms = Compose([
    RandomCrop(),
    affine_transform,
    Resize((256, 256)),
])

# Convert the above data sequences into a PyTorch Dataset.
# Users can have their own way to create the Dataset, as long as for each iteration 
# it returns a tuple of arrays for a single image: (image, bboxes, labels).
ds = PTDataset(image_list, bboxes_list, labels_list, transforms=transforms)
```

# API

## Overview

There are three modules: the Numpy transformation tools are from `targetran.np`,
the TensorFlow transformation tools are from `targetran.tf`, and some general
helper utilities are from `targetran.utils`.

### Transformation classes

Each transformation class comes in a pair, with one operating 
on `np.ndarray` and the other on `tf.Tensor`. For the latter, the class names 
have a `TF*` prefix, e.g., `RandomRotate` and `TFRandomRotate`.

The transformation classes are callables that accept input parameters from 
a single image:

- `image` (`np.ndarray` or `tf.Tensor` of shape `(image_height, image_width, num_channels)`);
- `bboxes` (`np.ndarray` or `tf.Tensor` of shape `(num_bboxes_per_image, 4)`, can be empty);
- `labels` (`np.ndarray` or `tf.Tensor` of shape `(num_bboxes_per_image,)`, can be empty).

The return format is a tuple: `(image, bboxes, labels)`.

Please see the [data format](#data-format) section for usage instructions.

### Transformation functions

There are also a pure functional counterpart for each transformation class, 
e.g., `rotate` and `tf_rotate` for `np.ndarray` and `tf.Tensor`, 
to which one could provide exact transformation parameters.

The input format is `(image, bboxes, labels, ...)` where each function
expects different additional input parameters. The return format is still
`(image, bboxes, labels)`.


## Full list

`targetran.np`
- `CombineAffine`
- `RandomFlipLeftRight`
- `RandomFlipUpDown`
- `RandomRotate`
- `RandomShear`
- `RandomTranslate`
- `RandomCrop`
- `Resize`
- `flip_left_right`
- `flip_up_down`
- `rotate`
- `shear`
- `translate`
- `crop`

`targetran.tf`
- `TFCombineAffine`
- `TFRandomFlipLeftRight`
- `TFRandomFlipUpDown`
- `TFRandomRotate`
- `TFRandomShear`
- `TFRandomTranslate`
- `TFRandomCrop`
- `TFResize`
- `to_tf`
- `seqs_to_tf_dataset`
- `tf_flip_left_right`
- `tf_flip_up_down`
- `tf_rotate`
- `tf_shear`
- `tf_translate`
- `tf_crop`

`targetran.utils`
- `Compose`
- `collate_fn`
- `image_only`

# Manual

