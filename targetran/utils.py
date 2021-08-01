"""
Utilities.
"""

from typing import Sequence, Tuple

import numpy as np

from .np import RandomTransform


class Compose:

    def __init__(self, fns: Sequence[RandomTransform]) -> None:
        self.fns = fns

    def __call__(
            self,
            image: np.ndarray,
            bboxes: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for fn in self.fns:
            image, bboxes, labels = fn(image, bboxes, labels)
        return image, bboxes, labels
