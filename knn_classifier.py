# optimized_dw_nca/knn_classifier.py

from sklearn.neighbors import KNeighborsClassifier as SK_KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class DWNCA_KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A K-Nearest Neighbors (KNN) classifier using Distance-Weighted Neighborhood Components Analysis (DW-NCA) for dimensionality reduction.

    Parameters:
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for kneighbors queries.
    n_components : int, default=None
        Number of components to use for dimensionality reduction.
    weights : str or callable, default='uniform'
        Weight function used in prediction. Possible values:
        - 'uniform': uniform weights.
        - 'distance': weight points by the inverse of their distance.
        - A callable function: a user-defined function that takes distances and returns weights.

    Attributes:
    ----------
    estimator_ : SK_KNeighborsClassifier
        The underlying KNeighborsClassifier instance fitted with the transformed data.
    dw_nca_ : DistanceWeightedNCA
        The DistanceWeightedNCA instance used for dimensionality reduction.
    """

    def __init__(self, n_neighbors=5, n_components=None, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the DW-NCA + KNN model to the data.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.

        Returns:
        -------
        self : object
            Returns self.
        """
        from optimized_dw_nca.distance_weighted_nca import DistanceWeightedNCA

        # Initialize DistanceWeightedNCA and transform data
        self.dw_nca_ = DistanceWeightedNCA(n_components=self.n_components)
        X_transformed = self.dw_nca_.fit_transform(X, y)

        # Initialize and fit KNeighborsClassifier
        self.estimator_ = SK_KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
        self.estimator_.fit(X_transformed, y)

        return self

    def predict(self, X):
        """
        Predict the labels for the given data using the fitted DW-NCA + KNN model.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to predict.

        Returns:
        -------
        y_pred : array, shape (n_samples,)
            Predicted labels.
        """
        X_transformed = self.dw_nca_.transform(X)
        return self.estimator_.predict(X_transformed)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the given data using the fitted DW-NCA + KNN model.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to predict.

        Returns:
        -------
        proba : array, shape (n_samples, n_classes)
            Class probabilities of the input samples.
        """
        X_transformed = self.dw_nca_.transform(X)
        return self.estimator_.predict_proba(X_transformed)
