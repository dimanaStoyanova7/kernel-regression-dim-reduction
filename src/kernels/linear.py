# src/kernels/linear.py

import numpy as np
from .base import Kernel


class LinearKernel(Kernel):
    """
    Linear kernel: k(x, z) = x^T z

    Note: if used in KRR it is equivalent to standard ridge regression.
    """

    def __call__(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Compute the linear kernel matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples_X, n_features)
        Z : np.ndarray of shape (n_samples_Z, n_features)

        Returns
        -------
        K : np.ndarray of shape (n_samples_X, n_samples_Z)
        """
        return X @ Z.T
