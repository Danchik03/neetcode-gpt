import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        num_layers = len(weights)
        curr = x
        for i in range(num_layers):
            curr = curr @ weights[i] + biases[i]   
            if i < num_layers - 1:
                curr = np.maximum(0, curr)
        return np.round(curr, 5)
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        pass
