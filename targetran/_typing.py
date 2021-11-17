"""
Typing utilities.
"""

from typing import Any, TypeVar

import numpy as np  # type: ignore

# This roughly means anything that is ndarray-like.
# Still looking for a better way.
T = TypeVar("T", np.ndarray, Any)
