#!/usr/bin/env python3
"""
TensorFlow Dataset example.
"""

from typing import Dict, List, Tuple

import os
import glob
import json

import cv2
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

from targetran.tf import (
    seqs_to_tf_dataset,
    TFCombineAffine,
    TFRandomFlipLeftRight,
    TFRandomFlipUpDown,
    TFRandomRotate,
    TFRandomShear,
    TFRandomCrop,
    TFRandomTranslate
)

AUTO = tf.data.AUTOTUNE


def load_images() -> Dict[str, np.ndarray]:
    """
    Users may do it differently depending on the data.
    """
    image_paths = glob.glob("./images/*.jpg")

    image_dict: Dict[str, np.ndarray] = {}
    for image_path in image_paths:
        image: np.ndarray = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        basename = os.path.basename(image_path)
        image_id = basename.split(".")[0]

        image_dict[image_id] = image

    return image_dict


def load_annotations() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Users may do it differently depending on the data.
    """
    with open("./annotations.json", "rb") as f:
        data = json.load(f)

    data_dict: Dict[str, Dict[str, np.ndarray]] = {}
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
        image_dict: Dict[str, np.ndarray],
        annotation_dict: Dict[str, Dict[str, np.ndarray]]
) -> tf.data.Dataset:
    """
    The main point is the item order of each list must match correspondingly.
    """
    image_list: List[np.ndarray] = []
    bboxes_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    for image_id, image in image_dict.items():
        image_list.append(image)
        bboxes_list.append(annotation_dict[image_id]["bboxes"])
        labels_list.append(annotation_dict[image_id]["labels"])

    return seqs_to_tf_dataset(image_list, bboxes_list, labels_list)


def plot(
        ds: tf.data.Dataset,
        num_rows: int,
        num_cols: int,
        figure_size_inches: Tuple[float, float] = (9.5, 7.5)
) -> None:
    """
    Plot samples of image, bboxes, and the corresponding labels.
    """
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figure_size_inches)

    for n, sample in enumerate(ds.take(num_rows * num_cols)):

        image, bboxes, labels = [tensor.numpy() for tensor in sample]
        image = image.astype(np.int32)

        for bbox, label in zip(bboxes, labels):
            x_min, y_min, width, height = [int(v) for v in bbox]

            cv2.rectangle(
                image, (x_min, y_min), (x_min + width, y_min + height),
                color=(0, 0, 255),  # Blue.
                thickness=3
            )
            cv2.putText(
                image, str(int(label)), (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                color=(0, 0, 255), thickness=2
            )

        if num_rows == 1 or num_cols == 1:
            ax = axes[n]
        else:
            ax = axes[n % num_rows][n % num_cols]

        ax.imshow(image)
        ax.set_axis_off()
        fig.set_tight_layout(True)

    plt.show()


def main() -> None:
    ds = make_tf_dataset(load_images(), load_annotations())

    affine_transform = TFCombineAffine([
        TFRandomRotate(),
        TFRandomShear(),
        TFRandomFlipLeftRight(probability=0.5, seed=0),
        TFRandomFlipUpDown(probability=0.5, seed=0),
        TFRandomTranslate(),
    ], probability=1.0, seed=0)

    # The `repeat` call is for re-using the same samples in this illustration.
    ds = ds \
        .repeat() \
        .map(TFRandomCrop(probability=1.0, seed=0), num_parallel_calls=AUTO) \
        .map(affine_transform, num_parallel_calls=AUTO) \

    plot(ds, num_rows=2, num_cols=3)


if __name__ == "__main__":
    main()
