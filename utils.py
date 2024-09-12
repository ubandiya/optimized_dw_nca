# optimized_dw_nca/utils.py

import numpy as np

def pairwise_squared_distances(X):
    """
    Compute the pairwise squared Euclidean distances between samples in X.

    Parameters:
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    Returns:
    -------
    distances : array, shape (n_samples, n_samples)
        Pairwise squared distances between samples.
    """
    sum_X = np.sum(X**2, axis=1)
    return sum_X[:, np.newaxis] + sum_X - 2 * np.dot(X, X.T)

def initialize_components(n_components, n_features):
    """
    Initialize the transformation matrix with random values.

    Parameters:
    ----------
    n_components : int
        Number of components.
    n_features : int
        Number of features in the data.

    Returns:
    -------
    A : array, shape (n_components, n_features)
        Initialized transformation matrix.
    """
    return np.random.randn(n_components, n_features)

def check_positive(value, name):
    """
    Check if a given value is positive.

    Parameters:
    ----------
    value : int or float
        The value to check.
    name : str
        The name of the value, used for error messages.

    Raises:
    ------
    ValueError
        If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive. Got {value}.")
