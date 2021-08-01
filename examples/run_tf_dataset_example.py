#!/usr/bin/env python3
"""
TensorFlow Dataset example.
"""

from typing import Dict, List

import os
import glob
import json

import cv2
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf

from targetran.tf import (
    np_to_tf,
    TFRandomRotate,
    TFRandomShear,
    TFRandomCrop,
    TFRandomTranslate
)


def load_images() -> Dict[str, np.ndarray]:

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

    image_list: List[np.ndarray] = []
    bboxes_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    for image_id, image in image_dict.items():

        image_list.append(image)
        bboxes_list.append(annotation_dict[image_id]["bboxes"])
        labels_list.append(annotation_dict[image_id]["labels"])

    tf_image_list, tf_bboxes_list, tf_labels_list = np_to_tf(
        image_list, bboxes_list, labels_list
    )

    # Tensors of different shapes can be included in a TF Dataset
    # as ragged-tensors.
    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_image_list)),
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_bboxes_list)),
        tf.data.Dataset.from_tensor_slices(tf.ragged.stack(tf_labels_list))
    ))
    # However, our transformations expect normal tensors, so the ragged-tensors
    # have to be first converted back to tensors during mapping. Therefore,
    # the whole point of using ragged-tensors is ONLY for building a Dataset...
    # Note that the label ragged-tensors are of rank-0, so they are implicitly
    # converted to tensors during mapping. Strange TF Dataset behaviour...
    ds = ds.map(lambda i, b, l: (i.to_tensor(), b.to_tensor(), l))

    return ds


def plot(ds: tf.data.Dataset, num_rows: int, num_cols: int) -> None:

    assert num_rows > 1 and num_cols > 1
    fig, axes = plt.subplots(num_rows, num_cols)

    for n, sample in enumerate(ds.take(num_rows * num_cols)):

        image, bboxes, labels = [tensor.numpy() for tensor in sample]

        for bbox, label in zip(bboxes, labels):

            image = image.astype(np.int32)
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

        axes[n % num_rows][n % num_cols].imshow(image)
        axes[n % num_rows][n % num_cols].set_axis_off()
        fig.set_tight_layout(True)

    plt.show()


def main() -> None:

    ds = make_tf_dataset(load_images(), load_annotations())

    ds = ds \
        .map(TFRandomCrop(probability=1.0)) \
        .map(TFRandomTranslate(probability=1.0)) \
        .map(TFRandomRotate(probability=1.0)) \
        .map(TFRandomShear(probability=1.0)) \
        .repeat()  # Re-using the same samples for illustration.

    plot(ds, num_rows=2, num_cols=3)


if __name__ == "__main__":
    main()
