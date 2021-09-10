#!/usr/bin/env python3
"""
PyTorch Dataset local example.
"""

import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

from targetran.np import (
    CombineAffine,
    RandomFlipLeftRight,
    RandomRotate,
    RandomShear,
    RandomCrop,
    RandomTranslate,
    Resize,
)
from targetran.utils import Compose, collate_fn


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


def make_pt_dataset(
        image_dict: Dict[str, np.ndarray],
        annotation_dict: Dict[str, Dict[str, np.ndarray]],
        transforms: Optional[Compose]
) -> Dataset:
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
    return PTDataset(image_seq, bboxes_seq, labels_seq, transforms)


def plot(
        ds: Dataset,
        num_rows: int,
        num_cols: int,
        figure_size_inches: Tuple[float, float] = (7.0, 4.5)
) -> None:
    """
    Plot samples of image, bboxes, and the corresponding labels.
    """
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figure_size_inches)

    for i in range(num_rows * num_cols):

        sample = ds[i % len(ds)]

        image, bboxes, labels = sample
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
    # The affine transformations can be combined for better performance.
    # Note that cropping and resizing are not affine.
    #
    # Option (1): each step has their own probability to be included.
    affine_transform = CombineAffine(
        [RandomRotate(probability=1.0),
         RandomShear(probability=1.0),
         RandomTranslate(probability=1.0),
         RandomFlipLeftRight(probability=0.5)],
        probability=1.0,
        seed=2
    )
    # Option (2): set the number of steps to be randomly chosen, e.g., 2.
    # This could be a better option because too many transformations may deform
    # the images too much.
    affine_transform = CombineAffine(
        [RandomRotate(),
         RandomShear(),
         RandomTranslate(),
         RandomFlipLeftRight()],
        num_selected_transforms=2,
        selected_probabilities=None,  # Default is None: uniform distribution.
        probability=1.0,
        seed=2
    )

    transforms = Compose([
        RandomCrop(probability=1.0, seed=1),
        affine_transform,
        Resize((640, 640)),
    ])

    ds = make_pt_dataset(load_images(), load_annotations(), transforms)

    plot(ds, num_rows=2, num_cols=3)

    # Example of batching with DataLoader and collate_fn.
    data_loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    for batch in data_loader:
        image_seq, bboxes_seq, labels_seq = batch
        print(f"transformed image-seq size: {len(image_seq)}")
        print(f"transformed bboxes-seq size: {len(bboxes_seq)}")
        print(f"transformed labels-seq size: {len(labels_seq)}")


if __name__ == "__main__":
    main()
