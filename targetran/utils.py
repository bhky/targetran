"""
Utilities.
"""

from typing import Any, Callable, Sequence, Tuple


class Compose:

    def __init__(self, fns: Sequence[Callable[..., Any]]) -> None:
        self.fns = fns

    def __call__(self, *args: Any) -> Any:
        for fn in self.fns:
            args = fn(*args)
        return args


def collate_fn(batch_: Sequence[Tuple[Any, ...]]) -> Tuple[Sequence[Any], ...]:
    return tuple(zip(*batch_))
