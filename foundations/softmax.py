import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z = z - max(z)
        thing = np.exp(z-max(z))
        ans = thing / np.sum(np.exp(z-max(z)) )
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        return np.round(ans, 4)
