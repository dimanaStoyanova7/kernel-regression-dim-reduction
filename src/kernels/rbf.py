# src/kernels/rbf.py

import numpy as np
from .base import Kernel


class RBFKernel(Kernel):
    """
    Radial Basis Function (Gaussian) kernel.

    k(x, z) = exp(-gamma * ||x - z||^2)
    """

    def __init__(self, gamma: float):
        """
        Parameters
        ----------
        gamma : float
            Controls the width of the Gaussian kernel.
            Must be positive.
        """
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        self.gamma = gamma

    def __call__(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Compute the RBF kernel matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples_X, n_features)
        Z : np.ndarray of shape (n_samples_Z, n_features)

        Returns
        -------
        K : np.ndarray of shape (n_samples_X, n_samples_Z)
        """

        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Z_norm = np.sum(Z ** 2, axis=1).reshape(1, -1)

        ## ∥x−z∥^2 = (x_transpose * x) + (z_transpose * z) − (2 * x_transpose * z)

        sq_dist = X_norm + Z_norm - 2 * X @ Z.T

        return np.exp(-self.gamma * sq_dist)
