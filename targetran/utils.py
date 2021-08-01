"""
Utilities.
"""

from typing import Any, Callable, Sequence


class Compose:

    def __init__(self, fns: Sequence[Callable[..., Any]]) -> None:
        self.fns = fns

    def __call__(self, *args: Any) -> Any:
        for fn in self.fns:
            args = fn(*args)
        return args
