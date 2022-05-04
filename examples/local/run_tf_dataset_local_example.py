#!/usr/bin/env python3
"""
TensorFlow Dataset local example.
"""
import glob
import json
import os
from typing import Dict, List, Tuple

import cv2
import matplotlib.pylab as plt
import numpy as np
import numpy.typing
import tensorflow as tf

from targetran.tf import (
    seqs_to_tf_dataset,
    TFCombineAffine,
    TFRandomFlipLeftRight,
    TFRandomRotate,
    TFRandomShear,
    TFRandomTranslate,
    TFRandomCrop,
    TFResize,
)

NDAnyArray = np.typing.NDArray[np.float_]


def load_images() -> Dict[str, NDAnyArray]:
    """
    Users may do it differently depending on the data.
    """
    image_paths = glob.glob("./images/*.jpg")

    image_dict: Dict[str, NDAnyArray] = {}
    for image_path in image_paths:
        image: NDAnyArray = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        basename = os.path.basename(image_path)
        image_id = basename.split(".")[0]

        image_dict[image_id] = image

    return image_dict


def load_annotations() -> Dict[str, Dict[str, NDAnyArray]]:
    """
    Users may do it differently depending on the data.
    """
    with open("./annotations.json", "rb") as f:
        data = json.load(f)

    data_dict: Dict[str, Dict[str, NDAnyArray]] = {}
    for image_item in data:

        image_id = image_item["image_id"]

        bboxes: List[List[int]] = []
        labels: List[int] = []
        for annotation in image_item["annotations"]:
            bboxes.append([
                annotation["top_left_x"],
                annotation["top_left_y"],
                annotation["width"],
                annotation["height"]
            ])
            labels.append(annotation["label"])

        data_dict[image_id] = {
            "bboxes": np.array(bboxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.float32)
        }

    return data_dict


def make_tf_dataset(
        image_dict: Dict[str, NDAnyArray],
        annotation_dict: Dict[str, Dict[str, NDAnyArray]]
) -> tf.data.Dataset:
    """
    Users may do it differently depending on the data.
    The main point is the item order of each sequence must match accordingly.
    """
    image_seq = [image for image in image_dict.values()]
    bboxes_seq = [
        annotation_dict[image_id]["bboxes"] for image_id in image_dict.keys()
    ]
    labels_seq = [
        annotation_dict[image_id]["labels"] for image_id in image_dict.keys()
    ]
    return seqs_to_tf_dataset(image_seq, bboxes_seq, labels_seq)


def plot(
        ds: tf.data.Dataset,
        num_rows: int,
        num_cols: int,
        figure_size_inches: Tuple[float, float] = (7.0, 4.5)
) -> None:
    """
    Plot samples of image, bboxes, and the corresponding labels.
    """
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figure_size_inches)

    for i, sample in enumerate(ds.take(num_rows * num_cols)):

        image, bboxes, labels = [tensor.numpy() for tensor in sample]
        image = image.astype(np.int32)

        for bbox, label in zip(bboxes, labels):
            x_min, y_min, width, height = [int(v) for v in bbox]

            cv2.rectangle(
                image, (x_min, y_min), (x_min + width, y_min + height),
                color=(0, 0, 255),  # Blue.
                thickness=2
            )
            cv2.putText(
                image, str(int(label)), (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                color=(0, 0, 255), thickness=2
            )

        if num_rows == 1 or num_cols == 1:
            ax = axes[i]
        else:
            ax = axes[i % num_rows][i % num_cols]

        ax.imshow(image)
        ax.set_axis_off()
        fig.set_tight_layout(True)

    plt.show()


def main() -> None:
    ds = make_tf_dataset(load_images(), load_annotations())

    # The affine transformations can be combined for better performance.
    # Note that cropping and resizing are not affine.
    #
    # Option (1): each step has their own probability to be included.
    affine_transform = TFCombineAffine(
        [TFRandomRotate(probability=1.0),
         TFRandomShear(probability=1.0),
         TFRandomTranslate(probability=1.0),
         TFRandomFlipLeftRight(probability=0.5)],
        probability=1.0,
        seed=2
    )
    # Option (2): set the number of steps to be randomly chosen, e.g., 2.
    # This could be a better option because too many transformations may deform
    # the images too much.
    affine_transform = TFCombineAffine(
        [TFRandomRotate(),
         TFRandomShear(),
         TFRandomTranslate(),
         TFRandomFlipLeftRight()],
        num_selected_transforms=2,
        selected_probabilities=None,  # Default is None: uniform distribution.
        probability=1.0,
        seed=2
    )

    # The `repeat` call here is only for re-using samples in this illustration.
    ds = ds \
        .repeat() \
        .map(TFRandomCrop(probability=1.0, seed=1)) \
        .map(affine_transform) \
        .map(TFResize((640, 640)))

    plot(ds, num_rows=2, num_cols=3)

    # Example of using the dataset with padded-batching.
    ds = ds.padded_batch(2, padding_values=np.nan)

    for batch in ds.take(5):
        image_batch, bboxes_batch, labels_batch = batch
        print("--------------")
        print(f"transformed image-batch shape: {image_batch.get_shape()}")
        print(f"transformed bboxes-batch shape: {bboxes_batch.get_shape()}")
        print(f"transformed labels-batch shape: {bboxes_batch.get_shape()}")


if __name__ == "__main__":
    main()
