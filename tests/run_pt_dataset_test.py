#!/usr/bin/env python3
"""
PyTorch Dataset test.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader

import targetran.np
from targetran.utils import Compose, collate_fn


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
            [99, 132, 20, 15],
            [340, 220, 31, 12],
        ], dtype=np.float32),
    ]

    labels_list = [
        np.array([0, 1], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([2, 3, 0], dtype=np.float32),
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
        targetran.np.RandomFlipUpDown(probability=1.0)
    ])

    print("-------- Random transform --------")

    ds = TestDataset(image_list, bboxes_list, labels_list, transforms)

    for sample in ds:
        image, bboxes, labels = sample
        print(f"transformed image shape: {image.shape}")
        print(f"transformed bboxes shape: {bboxes.shape}")
        print(f"transformed bboxes: {bboxes.tolist()}")
        print(f"transformed labels shape: {labels.shape}")
        print(f"transformed labels: {labels.tolist()}")
        print("=========")

    print("-------- Batching --------")

    data_loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    for batch in data_loader:
        image_tuple, bboxes_tuple, labels_tuple = batch
        print(f"transformed image batch size: {len(image_tuple)}")
        print(f"transformed bboxes batch size: {len(bboxes_tuple)}")
        print(f"transformed labels batch size: {len(labels_tuple)}")
        print(f"image shapes: {[i.shape for i in image_tuple]}")
        print(f"bboxes shapes: {[b.shape for b in bboxes_tuple]}")
        print(f"labels shapes: {[l.shape for l in labels_tuple]}")
        print("=========")


if __name__ == "__main__":
    main()
