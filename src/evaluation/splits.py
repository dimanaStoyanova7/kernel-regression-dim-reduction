import numpy as np
"""
    Returns indices for a single dev and test split.

    Parameters
    ----------
    n : int - total number of samples.
    test_size : float, default=0.2
    seed : int, default=42

    Returns
    -------
    dev_idx : np.ndarray of shape (n_dev,) - indices of samples assigned to the development (train+validation) set.
    test_idx : np.ndarray of shape (n_test,) - indices of samples assigned to the test set.
    """

def train_test_split_indices(n, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    dev_idx = idx[n_test:]
    return dev_idx, test_idx

"""
    Generator function that yields train and validation indices for k-fold cross-validation.

    Parameters
    ----------
    n : int - total number of samples in dev set
    k : int, default=5 - number of cross-validation folds.
    seed : int, default=42

    Yield
    ------
    train_idx : np.ndarray of shape (n_train,) - indices of samples used for training in the current fold.
    val_idx : np.ndarray of shape (n_val,) - indices of samples used for validation in the current fold.
    """

def kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        )
        yield train_idx, val_idx
