#!/usr/bin/env python3
"""
PyTorch Dataset test.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

import targetran.np
from targetran.utils import Compose


def make_np_data() -> Tuple[Sequence[np.ndarray],
                            Sequence[np.ndarray],
                            Sequence[np.ndarray]]:

    image_list = [np.random.rand(480, 512, 3) for _ in range(3)]

    bboxes_list = [
        np.array([
            [214, 223, 10, 11],
            [345, 230, 21, 9],
        ], dtype=np.float32),
        np.array([], dtype=np.float32).reshape(-1, 4),
        np.array([
            [104, 151, 22, 10],
        ], dtype=np.float32),
    ]

    labels_list = [
        np.array([0, 1], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([2], dtype=np.float32),
    ]

    return image_list, bboxes_list, labels_list


class TestDataset(Dataset):

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


def main() -> None:

    image_list, bboxes_list, labels_list = make_np_data()

    transforms = Compose([
        targetran.np.RandomRotate(probability=1.0),
        targetran.np.RandomCrop(probability=1.0),
        targetran.np.RandomFlipUpDown(probability=1.0)
    ])

    ds = TestDataset(image_list, bboxes_list, labels_list, transforms)

    for sample in ds:
        image, bboxes, labels = sample
        print(f"transformed image shape: {image.shape}")
        print(f"transformed bboxes shape: {bboxes.shape}")
        print(f"transformed bboxes: {bboxes}")
        print(f"transformed labels shape: {labels.shape}")
        print(f"transformed labels: {labels}")


if __name__ == "__main__":
    main()
