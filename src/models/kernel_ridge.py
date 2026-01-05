# src/models/kernel_ridge.py

import numpy as np
from src.kernels.base import Kernel

class KernelRidge:
    def __init__(self, kernel: Kernel, lambd: float):
        """
        Kernel Ridge Regression Model

        Parameters
        ----------
        kernel: Kernel object
        lambd: float
            regularization parameter
        """
        self.kernel = kernel
        self.lambd = lambd

        self.X_train = None
        self.alpha = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Set the training data

        Parameters
        ----------
        X : np.ndarray
            Training data
        y : np.ndarray
            Training labels
        """
        self.X_train = X
        K = self.kernel(X, X)
        n = K.shape[0]
        self.alpha = np.linalg.solve(K + self.lambd * np.eye(n), y)
        return self
    
    def predict(self, X_test: np.ndarray):
        """
        For each data sample in X_test, solve a weighted least squares problem

        Parameters
        ----------
        X_test : np.ndarray
            Testing data

        Returns
        -------
        Array of predictions
        """
        K = self.kernel(self.X_train, X_test)
        return K.T@self.alpha