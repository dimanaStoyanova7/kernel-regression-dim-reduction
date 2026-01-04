# src/kernels/polynomial.py

import numpy as np
from .base import Kernel


class PolynomialKernel(Kernel):
    """
    Polynomial kernel.

    k(x, z) = (gamma * x^T z + coef0) ^ degree
    """

    def __init__(self, degree: int, gamma: float = 1.0, coef0: float = 0.0):
        """
        Parameters
        ----------
        degree : int
            Degree of the polynomial.
        gamma : float, optional
            Scaling factor for the dot product.
        coef0 : float, optional
            Independent term (bias).
        """
        if degree <= 0:
            raise ValueError("degree must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def __call__(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Compute the polynomial kernel matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples_X, n_features)
        Z : np.ndarray of shape (n_samples_Z, n_features)

        Returns
        -------
        K : np.ndarray of shape (n_samples_X, n_samples_Z)
        """

        # x^T z -> pairwise dot products
        dot = X @ Z.T

        # (gamma * x^T z + coef0) ^ degree
        return (self.gamma * dot + self.coef0) ** self.degree
