"""
Typing utilities.
"""

from typing import Any

import numpy as np
import numpy.typing

ArrayLike = np.typing.ArrayLike
NDAnyArray = np.typing.NDArray[Any]

# T is treated semantically as "NPArray or tf.Tensor" in this library.
# However, there is currently no way to express this accurately in the
# typing system. Hence, it is assumed here that tf.Tensor is compatible
# with NPArray, which is kind of true. Hope there will be a better way.
T = NDAnyArray
