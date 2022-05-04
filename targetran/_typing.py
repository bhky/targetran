"""
Typing utilities.
"""
from typing import Any

import numpy as np
import numpy.typing

ArrayLike = np.typing.ArrayLike
NDAnyArray = np.typing.NDArray[Any]
NDBoolArray = np.typing.NDArray[np.bool_]
NDFloatArray = np.typing.NDArray[np.float_]
NDIntArray = np.typing.NDArray[np.int_]

# T is treated semantically as "NDAnyArray or tf.Tensor" in this library.
# However, there is currently no way to express this accurately in the
# typing system. Hence, it is assumed here that tf.Tensor is compatible
# with NDAnyArray, which is kind of true. Hope there will be a better way.
T = NDAnyArray
