#!/usr/bin/env python3
"""
Usage example of Targetran with TensorFlow Dataset, using data from Kaggle:
https://www.kaggle.com/c/global-wheat-detection/data

This example is to be run on a Kaggle Notebook, with the above dataset added.
"""

import json
import os
from typing import Dict, Optional

# Needed for the Kaggle Notebook.
os.system("pip install -U targetran")

import cv2
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from targetran.tf import (
    seqs_to_tf_dataset,
    TFCombineAffine,
    TFRandomFlipLeftRight,
    TFRandomRotate,
    TFRandomShear,
    TFRandomCrop,
    TFRandomTranslate,
    TFResize,
)

# This will be the data path when you use "Add Data" on the right panel
# of a Kaggle Notebook.
DATA_DIR = "/kaggle/input/global-wheat-detection"


def make_df(
        num_images: Optional[int] = None,
        data_dir: str = DATA_DIR
) -> pd.DataFrame:
    """
    Read and arrange data from train.csv and the train image directory.
    """
    train_csv_path = os.path.join(data_dir, "train.csv")
    train_images_dir = os.path.join(data_dir, "train")

    # Add image paths to df.
    df_wheat = pd.read_csv(train_csv_path)
    df_wheat["image_path"] = df_wheat["image_id"].apply(
        lambda image_id: os.path.join(train_images_dir, f"{image_id}.jpg")
    )

    df_wheat = df_wheat[["image_id", "image_path", "bbox"]]
    wheat_image_id_set = set(df_wheat["image_id"])
    print(f"Total number of images with wheat: {len(wheat_image_id_set)}")

    # Only select the needed number of images if given.
    if num_images is not None:
        image_ids = list(wheat_image_id_set)[:num_images]
        df_wheat = df_wheat[df_wheat["image_id"].isin(image_ids)]
        wheat_image_id_set = set(df_wheat["image_id"])
        print(f"Selected number of images: {len(wheat_image_id_set)}")

    # Each image has multiple rows by default, with each row giving one bbox.
    # Here we group the bboxes so that each image has one row only.
    df_wheat["bbox"] = df_wheat["bbox"].apply(json.loads)
    df_wheat_grouped = df_wheat.groupby("image_id", sort=False).agg(
        {"image_path": "first",
         "bbox": lambda s: s.tolist()}
    ).rename(columns={"bbox": "bboxes"}).reset_index()

    df_wheat_grouped = df_wheat_grouped[["image_id", "image_path", "bboxes"]]

    return df_wheat_grouped


def load_images(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    This way may be a bit clumsy, but looks clearer as an example.
    """
    image_ids = df["image_id"].tolist()
    image_paths = df["image_path"].tolist()

    image_dict: Dict[str, np.ndarray] = {}
    for image_id, image_path in zip(image_ids, image_paths):
        if image_id in image_dict:
            continue
        image: np.ndarray = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_dict[image_id] = image

    return image_dict


def load_annotations(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """
    This way may be a bit clumsy, but looks clearer as an example.
    """
    image_ids = df["image_id"].tolist()
    df = df.set_index("image_id")

    data_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for image_id in image_ids:
        bboxes = df.loc[image_id, "bboxes"]
        labels = [1] * len(bboxes)
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


def save_plots(ds: tf.data.Dataset, num_images: int) -> None:
    """
    Plot samples of image, bboxes, and the corresponding labels.
    """
    for i, sample in enumerate(ds.take(num_images)):

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

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)
        ax.set_axis_off()
        fig.set_tight_layout(True)
        fig.savefig(f"figure_{i}.png")


def main() -> None:
    num_images = 6
    df = make_df(num_images=num_images)
    ds = make_tf_dataset(load_images(df), load_annotations(df))

    # The affine transformations can be combined for better performance.
    # Note that cropping and resizing are not affine.
    affine_transform = TFCombineAffine([
        TFRandomRotate(probability=1.0),
        TFRandomShear(probability=1.0),
        TFRandomTranslate(probability=1.0),
        TFRandomFlipLeftRight(probability=0.5),
    ], probability=1.0, seed=0)

    ds = ds \
        .map(TFRandomCrop(probability=1.0, seed=1)) \
        .map(affine_transform) \
        .map(TFResize((960, 960)))

    save_plots(ds, num_images=num_images)


if __name__ == "__main__":
    main()
