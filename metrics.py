# optimized_dw_nca/metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class PerformanceMetrics:
    """
    A class to compute and store various performance metrics for classification tasks.

    Parameters:
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_pred : array-like, shape (n_samples,)
        Predicted labels.

    Attributes:
    ----------
    accuracy : float
        Accuracy score of the predictions.
    precision : float
        Precision score of the predictions.
    recall : float
        Recall score of the predictions.
    f1 : float
        F1 score of the predictions.
    """
    
    def __init__(self, y_true, y_pred):
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, average='weighted')
        self.recall = recall_score(y_true, y_pred, average='weighted')
        self.f1 = f1_score(y_true, y_pred, average='weighted')

    def __repr__(self):
        return (f"PerformanceMetrics(accuracy={self.accuracy:.4f}, "
                f"precision={self.precision:.4f}, "
                f"recall={self.recall:.4f}, "
                f"f1={self.f1:.4f})")

    def summary(self):
        """
        Return a summary of performance metrics as a dictionary.

        Returns:
        -------
        metrics_summary : dict
            Dictionary containing accuracy, precision, recall, and F1 score.
        """
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1
        }
