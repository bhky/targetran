"""
Typing utilities.
"""

from typing import Any

import numpy as np
import numpy.typing

NPArray = np.typing.NDArray[Any]

# T is treated as "NPArray or tf.Tensor" in this library.
# It is assumed that tf.Tensor is kind of compatible with NPArray,
# but still the usage of T and NPArray should be distinguished.
# Hope there will be a better way to do this in the future.
T = NPArray
