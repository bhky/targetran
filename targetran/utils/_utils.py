"""
Utilities.
"""

from typing import Any, Callable, Sequence, Tuple, TypeVar

T = TypeVar("T")


class Compose:

    def __init__(self, fns: Sequence[Callable[..., Any]]) -> None:
        self.fns = fns

    def __call__(self, *args: Any) -> Any:
        for fn in self.fns:
            args = fn(*args)
        return args


def collate_fn(batch: Sequence[Tuple[Any, ...]]) -> Tuple[Sequence[Any], ...]:
    return tuple(zip(*batch))


def image_only(
        tran_fn: Callable[[T, Any, Any], Tuple[T, T, T]]
) -> Callable[[T, T], Tuple[T, T]]:
    # Only the image will be transformed.
    def fn(image: T, *args: Any) -> Any:
        transformed_image = tran_fn(image, [], [])[0]
        if len(args) == 0:
            return transformed_image
        return (transformed_image, *args)

    return fn
