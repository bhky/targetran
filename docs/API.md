# Targetran API

## Table of contents
- [Overview](#overview)
  - [Transformation classes](#transformation-classes)
  - [Transformation functions](#transformation-functions)
  - [Random seeds](#random-seeds)
- [Full list](#full-list)
- [Manual](#manual)

## Overview

There are three modules:
- `targetran.np` for NumPy transformation tools;
- `targetran.tf` for TensorFlow transformation tools;
- `targetran.utils` for general utilities.

Please see [here](../README.md#usage) for notations and usage instructions.


### Transformation classes

Each transformation class comes in a pair, with one operating 
on `NDFloatArray` and the other on `tf.Tensor`. For the latter, the class names 
have a `TF*` prefix, e.g., `RandomRotate` and `TFRandomRotate`, respectively.

Pure TensorFlow ops are used for implementing the `TF*` tools,
which is why they could work smoothly with a TensorFlow Dataset on TPU.

The transformation classes are callables that accept input parameters from 
a single sample consists of:

- `image` (`NDFloatArray` or `tf.Tensor` of shape `(height, width, num_channels)`);
- `bboxes` (`NDFloatArray` or `tf.Tensor` of shape `(num_bboxes_per_image, 4)`, can be empty), 
  where each row is `[top_left_x, top_left_y, bbox_width, bbox_height]`;
- `labels` (`NDFloatArray` or `tf.Tensor` of shape `(num_bboxes_per_image,)`, can be empty).

The return format is a tuple: `(image, bboxes, labels)`.

### Transformation functions

Each transformation class also has a pure functional counterpart, 
e.g., `rotate` and `tf_rotate` for `NDFloatArray` and `tf.Tensor` 
(with `tf_*` prefix), 
to which one could provide exact transformation parameters.

The input format is `(image, bboxes, labels, ...)` where each function
expects different additional input parameters. The return format is still
`(image, bboxes, labels)`.

### Random seeds

It is worth to note that, while each `*Random*` class has a `seed` option 
for setting an individual random seed, it is usually enough to just set the
seed for Python, NumPy, and/or TensorFlow for deterministic operations.
That means, it is not needed to set the individual seeds if something like 
the following has been done:
```python
import random
import numpy as np
import tensorflow as tf
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
```
See [here](https://www.tensorflow.org/api_docs/python/tf/keras/utils/set_random_seed)
for more details.

## Full list

`targetran.np`
- [`RandomFlipLeftRight`](#randomflipleftright-tfrandomflipleftright)
- [`RandomFlipUpDown`](#randomflipupdown-tfrandomflipupdown)
- [`RandomRotate`](#randomrotate-tfrandomrotate)
- [`RandomShear`](#randomshear-tfrandomshear)
- [`RandomTranslate`](#randomtranslate-tfrandomtranslate)
- [`CombineAffine`](#combineaffine-tfcombineaffine)
- [`RandomCrop`](#randomcrop-tfrandomcrop)
- [`Resize`](#resize-tfresize)
- [`flip_left_right`](#flip_left_right-tf_flip_left_right)
- [`flip_up_down`](#flip_up_down-tf_flip_up_down)
- [`rotate`](#rotate-tf_rotate)
- [`shear`](#shear-tf_shear)
- [`translate`](#translate-tf_translate)
- [`crop`](#crop-tf_crop)
- [`resize`](#resize-tf_resize)

`targetran.tf`
- [`TFRandomFlipLeftRight`](#randomflipleftright-tfrandomflipleftright)
- [`TFRandomFlipUpDown`](#randomflipupdown-tfrandomflipupdown)
- [`TFRandomRotate`](#randomrotate-tfrandomrotate)
- [`TFRandomShear`](#randomshear-tfrandomshear)
- [`TFRandomTranslate`](#randomtranslate-tfrandomtranslate)
- [`TFCombineAffine`](#combineaffine-tfcombineaffine)
- [`TFRandomCrop`](#randomcrop-tfrandomcrop)
- [`TFResize`](#resize-tfresize)
- [`to_tf`](#to_tf)
- [`seqs_to_tf_dataset`](#seqs_to_tf_dataset)
- [`tf_flip_left_right`](#flip_left_right-tf_flip_left_right)
- [`tf_flip_up_down`](#flip_up_down-tf_flip_up_down)
- [`tf_rotate`](#rotate-tf_rotate)
- [`tf_shear`](#shear-tf_shear)
- [`tf_translate`](#translate-tf_translate)
- [`tf_crop`](#crop-tf_crop)
- [`tf_resize`](#resize-tf_resize)

`targetran.utils`
- [`Interpolation`](#interpolation)
- [`Compose`](#compose)
- [`collate_fn`](#collate_fn)
- [`image_only`](#image_only)

## Manual

### `RandomFlipLeftRight`, `TFRandomFlipLeftRight`
Randomly flip the input image horizontally (left to right).
- `__init__` parameters
  - `probability` (`float`, default `0.5`): Probability to apply the transformation.
  - `seed` (`Optional[int]`, default `None`): Random seed.
- `__call__` parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
- `__call__` returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `RandomFlipUpDown`, `TFRandomFlipUpDown`
Randomly flip the input image vertically (up to down).
- `__init__` parameters
  - `probability` (`float`, default `0.5`): Probability to apply the transformation.
  - `seed` (`Optional[int]`, default `None`): Random seed.
- `__call__` parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
- `__call__` returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `RandomRotate`, `TFRandomRotate`
Randomly rotate the input image about the centre.
- `__init__` parameters
  - `angle_deg_range` (`Tuple[float, float]`, default `(-15.0, 15.0)`):
    The lower (inclusive) and upper (exclusive, if value differs from lower) limits of the rotation angle in degrees.
    Positive values means anti-clockwise, and vice versa.
  - `interpolation` (`Interpolation` enum from `targetran.utils`, default `Interpolation.BILINEAR`):
    Interpolation mode. Only `Interpolation.BILINEAR` and `Interpolation.NEAREST` are supported.
  - `fill_value` (`float`, default `0.0`): Value to be filled outside the image boundaries.
  - `probability` (`float`, default `0.9`): Probability to apply the transformation.
  - `seed` (`Optional[int]`, default `None`): Random seed.
- `__call__` parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
- `__call__` returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `RandomShear`, `TFRandomShear`
Randomly shear the input image horizontally about the centre.
- `__init__` parameters
  - `angle_deg_range` (`Tuple[float, float]`, default `(-10.0, 10.0)`):
    The lower (inclusive) and upper (exclusive, if value differs from lower) limits of the shear angle in degrees.
    Positive values means anti-clockwise, and vice versa.
    Both values should be greater than `-90.0` and less than `90.0`.
  - `interpolation` (`Interpolation` enum from `targetran.utils`, default `Interpolation.BILINEAR`):
    Interpolation mode. Only `Interpolation.BILINEAR` and `Interpolation.NEAREST` are supported.
  - `fill_value` (`float`, default `0.0`): Value to be filled outside the image boundaries.
  - `probability` (`float`, default `0.9`): Probability to apply the transformation.
  - `seed` (`Optional[int]`, default `None`): Random seed.
- `__call__` parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
- `__call__` returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.
  
### `RandomTranslate`, `TFRandomTranslate`
Randomly translate the input image.
- `__init__` parameters
  - `translate_height_fraction_range` (`Tuple[float, float]`, default `(-0.1, 0.1)`):
    The lower (inclusive) and upper (exclusive, if value differs from lower) limits of the vertical translation, 
    given as fractions of the image height. 
    Both values should be greater than `-1.0` and less than `1.0`.
  - `translate_width_fraction_range` (`Tuple[float, float]`, default `(-0.1, 0.1)`):
    The lower (inclusive) and upper (exclusive, if value differs from lower) limits of the horizontal translation, 
    given as fractions of the image width. 
    Both values should be greater than `-1.0` and less than `1.0`.
  - `interpolation` (`Interpolation` enum from `targetran.utils`, default `Interpolation.BILINEAR`):
    Interpolation mode. Only `Interpolation.BILINEAR` and `Interpolation.NEAREST` are supported.
  - `fill_value` (`float`, default `0.0`): Value to be filled outside the image boundaries.
  - `probability` (`float`, default `0.9`): Probability to apply the transformation.
  - `seed` (`Optional[int]`, default `None`): Random seed.
- `__call__` parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
- `__call__` returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `CombineAffine`, `TFCombineAffine`
Combine the random affine transformations to improve performance.
- `__init__` parameters
  - `transforms` (Sequence of affine transform class objects): Accepted options are from below.
    - `RandomFlipLeftRight`/`TFRandomFlipLeftRight`
    - `RandomFlipUpDown`/`TFRandomFlipUpDown`
    - `RandomRotate`/`TFRandomRotate`
    - `RandomShear`/`TFRandomShear`
    - `RandomTranslate`/`TFRandomTranslate`
  - `num_selected_transforms` (`Optional[int]`, default `None`): 
    Number of transformation steps in `transforms` to be randomly selected.
    If `None`, whether a step will be selected is determined by each `probability` given,
    i.e., the number of selected steps could vary in different calls.
    See the [example scripts](../examples/local).
  - `selected_probabilities` (`Optional[List[float]]`, default `None`):
    The selection probabilities associated with each step in `transforms`. 
    If `None`, a uniform distribution over all steps is assumed. 
    Only valid if `num_selected_transforms` is an `int`.
  - `keep_order` (`bool`, default `False`): If `True`, the selected transformation steps 
    will be performed according to the given order in `transforms`, otherwise the steps
    will be shuffled.
  - `interpolation` (`Interpolation` enum from `targetran.utils`, default `Interpolation.BILINEAR`):
    Interpolation mode. Only `Interpolation.BILINEAR` and `Interpolation.NEAREST` are supported.
  - `fill_value` (`float`, default `0.0`): Value to be filled outside the image boundaries.
    Note that this value will override those in the individual steps.
  - `probability` (`float`, default `1.0`): Probability to apply the combined transformation.
  - `seed` (`Optional[int]`, default `None`): Random seed.
- `__call__` parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
- `__call__` returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `RandomCrop`, `TFRandomCrop`
Get a random crop of the input image.
- `__init__` parameters
  - `crop_height_fraction_range` (`Tuple[float, float]`, default `(0.8, 0.9)`):
    The lower (inclusive) and upper (exclusive, if value differs from lower) limits of the image crop height, 
    given as fractions of the image height. 
    Both values should be greater than `0.0` and less than `1.0`.
  - `crop_width_fraction_range` (`Tuple[float, float]`, default `(0.8, 0.9)`):
    The lower (inclusive) and upper (exclusive, if value differs from lower) limits of the image crop width, 
    given as fractions of the image width. 
    Both values should be greater than `0.0` and less than `1.0`.
  - `probability` (`float`, default `0.9`): Probability to apply the transformation.
  - `seed` (`Optional[int]`, default `None`): Random seed.
- `__call__` parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
- `__call__` returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `Resize`, `TFResize`
Resize the input image.
- `__init__` parameters
  - `dest_size` (`Tuple[int, int]`): Destination image size given as (height, width).
  - `interpolation` (`Interpolation` enum from `targetran.utils`, default `Interpolation.BILINEAR`):
    Interpolation mode.
- `__call__` parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
- `__call__` returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `to_tf`
Convert array sequences to TensorFlow (eager) tensor sequences.
- Parameters
  - `image_seq`, `bboxes_seq`, `labels_seq`, 
    `image_seq_is_paths` (`boolean`, default `False`): 
    Please refer to the [data format](../README.md#data-format) and the 
    illustration for [TensorFlow Dataset](../README.md#tensorflow-dataset).
- Returns
  - Tuple of tensors: `(image_seq, bboxes_seq, labels_seq)`.

### `seqs_to_tf_dataset`
Convert array sequences to a TensorFlow Dataset.
- Parameters
  - `image_seq`, `bboxes_seq`, `labels_seq`,
    `image_seq_is_paths` (`boolean`, default `False`):
    Please refer to the [data format](../README.md#data-format) and the 
    illustration for [TensorFlow Dataset](../README.md#tensorflow-dataset).
- Returns
  - `tf.data.Dataset` instance.

### `flip_left_right`, `tf_flip_left_right`
Flip the input image horizontally (left to right).
- Parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
- Returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `flip_up_down`, `tf_flip_up_down`
Flip the input image vertically (up to down).
- Parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
- Returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `rotate`, `tf_rotate`
Rotate the input image about the centre.
- Parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
  - `angle_deg` (`float`): Rotation angle in degrees. 
    A positive value means anti-clockwise, and vice versa.
  - `interpolation` (`Interpolation` enum from `targetran.utils`, default `Interpolation.BILINEAR`):
    Interpolation mode. Only `Interpolation.BILINEAR` and `Interpolation.NEAREST` are supported.
  - `fill_value` (`float`, default `0.0`): Value to be filled outside the image boundaries.
- Returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `shear`, `tf_shear`
Shear the input image horizontally about the centre.
- Parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
  - `angle_deg` (`float`): 
    Shear angle in degrees, must be greater than `-90.0` and less than `90.0`. 
    A positive value means anti-clockwise, and vice versa.
  - `interpolation` (`Interpolation` enum from `targetran.utils`, default `Interpolation.BILINEAR`):
    Interpolation mode. Only `Interpolation.BILINEAR` and `Interpolation.NEAREST` are supported.
  - `fill_value` (`float`, default `0.0`): Value to be filled outside the image boundaries.
- Returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.
  
### `translate`, `tf_translate`
Translate the input image.
- Parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
  - `translate_height` (`int`): Vertical translation in pixels,
    with its absolute value smaller than the image height.
    A positive value means moving downwards, and vice versa.
  - `translate_width` (`int`): Horizontal translation in pixels.
    with its absolute value smaller than the image width.
    A positive value means moving rightwards, and vice versa.
  - `interpolation` (`Interpolation` enum from `targetran.utils`, default `Interpolation.BILINEAR`):
    Interpolation mode. Only `Interpolation.BILINEAR` and `Interpolation.NEAREST` are supported.
  - `fill_value` (`float`, default `0.0`): Value to be filled outside the image boundaries.
- Returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.
  
### `crop`, `tf_crop`
Get a crop of the input image.
- Parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
  - `offset_height` (`int`): Offset height of the image crop in pixels,
    must be greater than `0` and less than the image height.
  - `offset_width` (`int`): Offset width of the image crop in pixels,
    must be greater than `0` and less than the image width.
  - `crop_height` (`int`): 
    In pixels, naturally bounded by the image height.
  - `crop_width` (`int`): 
    In pixels, naturally bounded by the image width.
- Returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `resize`, `tf_resize`
Resize the input image. The same as `Resize`/`TFResize` but in functional form.
- Parameters
  - `image`, `bboxes`, `labels`: Please refer to the [overview](#overview).
  - `dest_size` (`Tuple[int, int]`): Destination image size given as (height, width).
  - `interpolation` (`Interpolation` enum from `targetran.utils`, default `Interpolation.BILINEAR`):
    Interpolation mode.
- Returns
  - Tuple of the transformed `(image`, `bboxes`, `labels)`.

### `Interpolation`
Enum class for interpolation modes in their usual definitions.
- `Interpolation.NEAREST`
- `Interpolation.BILINEAR`
- `Interpolation.BICUBIC` (for resizing only)
- `Interpolation.AREA` (for resizing only)

### `Compose`
Make a composition of the given callables.
- `__init__` parameters
  - `fns`: Sequence of callables that have the same input and return format, 
    e.g., the transformation classes.
- `__call__` parameters
  - Same format as the given callables.
- `__call__` returns
  - Same format as the given callables.

### `collate_fn`
To be used with the `DataLoader` from PyTorch for batching.
- Parameters
  - `batch`: Sequence of tuples.
- Returns
  - Tuple of sequences.

### `image_only`
Convert a transformation class to transform the input image only, mainly for 
[image classification](../README.md#image-classification).
- Parameters
  - `tran_fn`: Transformation class object.
- Returns
  - Callable that performs the given transformation only to the input image
    while returning other input parameters untouched.
