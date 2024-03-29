"""
Utilities.
"""
from enum import auto, Enum
from typing import Any, Callable, Sequence, Tuple, Union

from targetran._typing import T


class Interpolation(Enum):
    NEAREST = auto()
    BILINEAR = auto()
    BICUBIC = auto()
    AREA = auto()


class Compose:

    def __init__(self, fns: Sequence[Callable[..., Any]]) -> None:
        self.fns = fns

    def __call__(self, *args: Any) -> Any:
        # This is needed, otherwise *args will "unpack" an image array.
        if len(args) == 1:
            x = args[0]
            for fn in self.fns:
                x = fn(x)
            return x

        for fn in self.fns:
            args = fn(*args)
        return args


def collate_fn(batch: Sequence[Tuple[Any, ...]]) -> Tuple[Sequence[Any], ...]:
    return tuple(zip(*batch))


def image_only(
        tran_fn: Callable[[T, Any, Any], Tuple[T, T, T]]
) -> Callable[..., Union[T, Tuple[T, ...]]]:
    # Only the image will be transformed.
    def fn(image: T, *args: Any) -> Any:
        transformed_image = tran_fn(image, [], [])[0]
        if len(args) == 0:
            return transformed_image
        return (transformed_image, *args)  # The parentheses are needed.

    return fn
