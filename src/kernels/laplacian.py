# src/kernels/laplacian.py

import numpy as np
from .base import Kernel


class LaplacianKernel(Kernel):
    """
    Laplacian kernel.

    k(x, z) = exp(-gamma * ||x - z||_1)
    """

    def __init__(self, gamma: float):
        """
        Parameters
        ----------
        gamma : float
            Controls the width of the Laplacian kernel.
            Must be positive.
        """
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        self.gamma = gamma

    def __call__(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Compute the Laplacian kernel matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples_X, n_features)
        Z : np.ndarray of shape (n_samples_Z, n_features)

        Returns
        -------
        K : np.ndarray of shape (n_samples_X, n_samples_Z)
        """

        nX, d = X.shape
        nZ = Z.shape[0]

        # ||x - z||_1 = sum_i |x_i - z_i|
        # Avoid allocating (nX, nZ, d) by accumulating per-feature into (nX, nZ)
        l1_dist = np.zeros((nX, nZ), dtype=np.float64)

        for j in range(d):
            l1_dist += np.abs(X[:, j][:, None] - Z[:, j][None, :])

        return np.exp(-self.gamma * l1_dist)

