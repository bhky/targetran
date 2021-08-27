![logo](logo/targetran_logo.png)

# Motivation

[Data augmentation](https://en.wikipedia.org/wiki/Data_augmentation) 
is a technique commonly used for training machine learning models in the
computer vision field, where one can increase the amount of image data by
creating transformed copies of the original images.

In the object detection sub-field, the transformation has to be done also
to the target rectangular bounding-boxes. However, such functionality is not 
readily available in frameworks such as TensorFlow and PyTorch.

Here comes Targetran to fill the gap.

# What is Targetran?

- A data augmentation library to assist object detection or image classification 
  model training. 
- Has simple Python API to transform both the images and the target rectangular 
  bounding-boxes.
- Use dataset-idiomatic approach for TensorFlow and PyTorch.

![example](docs/example.png)

(Figure produced by the example code [here](examples/local/run_tf_dataset_local_example.py).
The author owns the photos.)

# Table of contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data format](#data-format)
  - [Design principles](#design-principles)
  - [TensorFlow Dataset](#tensorflow-dataset)
  - [PyTorch Dataset](#pytorch-dataset)
  - [Image classification](#image-classification)
  - [Examples](#examples)
- [API](#api)

# Installation

Python 3 is required.

The best way to install Targetran with its dependencies is from PyPI:
```shell
python3 -m pip install --upgrade targetran
```
Alternatively, to obtain the latest version from this repository:
```shell
git clone https://github.com/bhky/targetran.git
cd targetran
python3 -m pip install .
```

# Usage

## Data format

For object detection model training, which is the primary usage here, the following data are needed.
- `image_seq` (Sequence of `np.ndarray` or `tf.Tensor` of shape `(height, width, num_channels)`):
  - images in channel-last format;
  - image sizes can be different.
- `bboxes_seq` (Sequence of `np.ndarray` or `tf.Tensor` of shape `(num_bboxes_per_image, 4)`):
  - each `bboxes` array/tensor provides the bounding-boxes associated with an image;
  - each single bounding-box is given as `[top_left_x, top_left_y, bbox_width, bbox_height]`;
  - empty array/tensor means no bounding-boxes (and labels) for that image.
- `labels_seq` (Sequence of `np.ndarray` or `tf.Tensor` of shape `(num_bboxes_per_image,)`):
  - each `labels` array/tensor provides the bounding-box labels associated with an image;
  - empty array/tensor means no labels (and bounding-boxes) for that image.

Some dummy data are created below for illustration. Please note the required format.
```python
import numpy as np

# Each image could have different sizes, but they must follow the channel-last format, 
# i.e., (height, width, num_channels).
image_seq = [np.random.rand(480, 512, 3) for _ in range(3)]

# The bounding-boxes (bboxes) are given as a sequence of NumPy arrays (or TF tensors).
# Each array represents the bboxes for one corresponding image.
#
# Each bbox is given as [top_left_x, top_left_y, bbox_width, bbox_height].
# 
# In case an image has no bboxes, an empty array should be provided.
bboxes_seq = [
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

# Labels for the bboxes are also given as a sequence of NumPy arrays (or TF tensors).
# The number of bboxes and labels should match. An empty array indicates no bboxes/labels.
labels_seq = [
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

## TensorFlow Dataset

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
# it returns a tuple of tensors for a single sample: (image, bboxes, labels).
ds = seqs_to_tf_dataset(image_seq, bboxes_seq, labels_seq)

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
```python
# Batching:
# Since the array/tensor shape of each sample could be different, conventional
# way of batching may not work. Users will have to consider their own use cases.
# One possibly useful way is the padded-batch.
ds = ds.padded_batch(batch_size=2, padding_values=-1.0)
```

## PyTorch Dataset

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
    As per common practice, transforms are done on NumPy arrays.
    """
    
    def __init__(
            self,
            image_seq: Sequence[np.ndarray],
            bboxes_seq: Sequence[np.ndarray],
            labels_seq: Sequence[np.ndarray],
            transforms: Optional[Compose]
    ) -> None:
        self.image_seq = image_seq
        self.bboxes_seq = bboxes_seq
        self.labels_seq = labels_seq
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_seq)

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.transforms:
            return self.transforms(
                self.image_seq[idx],
                self.bboxes_seq[idx],
                self.labels_seq[idx]
            )
        return (
            self.image_seq[idx],
            self.bboxes_seq[idx],
            self.labels_seq[idx]
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
# it returns a tuple of arrays for a single sample: (image, bboxes, labels).
ds = PTDataset(image_seq, bboxes_seq, labels_seq, transforms=transforms)
```
```python
# Batching:
# In PyTorch, it is common to use a Dataset with a DataLoader, which provides
# batching functionality. However, since the array/tensor shape of each sample 
# could be different, the default batching may not work. Targetran provides
# a `collate_fn` that helps producing batches of (image_seq, bboxes_seq, labels_seq).
from torch.utils.data import DataLoader
from targetran.utils import collate_fn

data_loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
```

## Image classification

While the tools here are primarily designed for object detection tasks, they can 
also be used for image classification in which only the images are to be transformed,
e.g., given a dataset that returns `(image, label)` samples, or even only `image` samples. 
The `image_only` function can be used to convert a transformation class for this purpose.

If the dataset returns a tuple `(image, ...)` in each iteration, only the `image`
will be transformed, other parameters that followed such as `(..., label, weight)` 
will be returned untouched.

If the dataset returns `image` only (not a tuple), then only the transformed `image` will be returned. 
```python
from targetran.utils import image_only
```
```python
# TensorFlow.
ds = ds \
    .map(image_only(TFRandomCrop())) \
    .map(image_only(affine_transform)) \
    .map(image_only(TFResize((256, 256))))
```
```python
# PyTorch.
transforms = Compose([
    image_only(RandomCrop()),
    image_only(affine_transform),
    image_only(Resize((256, 256))),
])
ds = PTDataset(..., transforms=transforms)
```

## Examples

See [here](examples) for example codes.

# API

See [here](docs/API.md) for API details.
