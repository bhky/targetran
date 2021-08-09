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


def collate_fn(batch_: Sequence[Tuple[Any, ...]]) -> Tuple[Sequence[Any], ...]:
    return tuple(zip(*batch_))


def to_classification(
        tran_fn: Callable[[T, Any, Any], Tuple[T, T, T]]
) -> Callable[[T, T], Tuple[T, T]]:

    def fn(image: T, label: T) -> Tuple[T, T]:
        return tran_fn(image, [], [])[0], label

    return fn
