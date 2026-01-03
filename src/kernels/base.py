# src/kernels/base.py

from abc import ABC, abstractmethod
import numpy as np


class Kernel(ABC):
    """
    Abstract base class for all kernels.

    All kernels must implement __call__(X, Z) and return
    a kernel matrix of shape (n_samples_X, n_samples_Z).
    """

    @abstractmethod
    def __call__(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix K(X, Z).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples_X, n_features)
            First input matrix.
        Z : np.ndarray of shape (n_samples_Z, n_features)
            Second input matrix.

        Returns
        -------
        K : np.ndarray of shape (n_samples_X, n_samples_Z)
            Kernel matrix.
        """
        pass
