# src/models/local_regression.py

import numpy as np
import matplotlib.pyplot as plt
from src.kernels.base import Kernel

class LocalRegression:
    def __init__(self, kernel: Kernel):
        """
        Local Regression model.

        Parameters
        ----------
        kernel : Kernel object (Linear, Polynomial, or RBF)
            The kernel used to calculate weights for local points.
        """
        self.kernel = kernel
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Set the training data

        Parameters
        ----------
        X : Training data
        y : Training labels
        """
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X_test):
        """
        For each data sample in X_test, solve a weighted least squares problem

        Parameters
        ----------
        X_test : Testing data

        Returns
        -------
        Array of predictions
        """
        predictions = [self.predict_single(x) for x in X_test]
        return np.array(predictions)

    def predict_single(self, x_query):
        # Calculate the weights using the kernel
        weights = self.kernel(self.X_train, x_query.reshape(1, -1)).flatten()
        W = np.diag(weights)

        # We add a bias term (column of 1s) for local linear fit
        X_b = np.c_[np.ones(self.X_train.shape[0]), self.X_train]
        x_q_b = np.r_[1, x_query]

        # Solve Weighted Least Squares: (X^T W X) beta = X^T W y
        # We use a small ridge penalty (1e-10) to ensure matrix invertibility
        A = X_b.T @ W @ X_b + np.eye(X_b.shape[1]) * 1e-10
        b = X_b.T @ W @ self.y_train
        
        beta = np.linalg.solve(A, b)

        return x_q_b @ beta