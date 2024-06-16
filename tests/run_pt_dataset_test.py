#!/usr/bin/env python3
"""
PyTorch Dataset test.
"""
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing
from torch.utils.data import Dataset, DataLoader

import targetran.np
from targetran.utils import Compose, collate_fn

NDAnyArray = np.typing.NDArray[np.float_]


def make_np_data() -> Tuple[Sequence[NDAnyArray],
                            Sequence[NDAnyArray],
                            Sequence[NDAnyArray]]:
    image_seq = [np.random.rand(480, 512, 3) for _ in range(3)]

    bboxes_seq = [
        np.array([
            [214, 223, 10, 11],
            [345, 230, 21, 9],
        ]),
        np.array([]),
        np.array([
            [104, 151, 22, 10],
            [99, 132, 20, 15],
            [340, 220, 31, 12],
        ]),
    ]

    labels_seq = [
        np.array([0, 1]),
        np.array([]),
        np.array([2, 3, 0]),
    ]

    return image_seq, bboxes_seq, labels_seq


class PTDataset(Dataset):

    def __init__(
            self,
            image_seq: Sequence[NDAnyArray],
            bboxes_seq: Sequence[NDAnyArray],
            labels_seq: Sequence[NDAnyArray],
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
    ) -> Tuple[NDAnyArray, NDAnyArray, NDAnyArray]:
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


def main() -> None:
    image_seq, bboxes_seq, labels_seq = make_np_data()

    transforms = Compose([
        targetran.np.RandomRotate(probability=1.0),
        targetran.np.RandomShear(probability=1.0),
        targetran.np.RandomTranslate(probability=1.0),
        targetran.np.RandomFlipUpDown(probability=1.0),
        targetran.np.RandomFlipLeftRight(probability=1.0),
        targetran.np.RandomCrop(probability=1.0),
        targetran.np.Resize((256, 256)),
    ])

    print("-------- Random transform --------")

    ds = PTDataset(image_seq, bboxes_seq, labels_seq, transforms)

    for example in ds:
        image, bboxes, labels = example
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
