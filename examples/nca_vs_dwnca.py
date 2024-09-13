# examples/nca_vs_dwnca.py

import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from optimized_dw_nca import DistanceWeightedNCA
from optimized_dw_nca.knn_classifier import DWNCA_KNNClassifier
from optimized_dw_nca.metrics import PerformanceMetrics

def load_datasets():
    """
    Load multiple datasets for comparison.

    Returns:
    -------
    datasets : dict
        Dictionary with dataset names as keys and dataset objects as values.
    """
    datasets = {
        'Iris': load_iris(),
        'Wine': load_wine(),
        'Breast Cancer': load_breast_cancer()
    }
    return datasets

def compare_methods(X, y, test_size=0.3, n_components=None, n_neighbors=3, cv=5):
    """
    Compare traditional NCA and DW-NCA on a given dataset with tuning.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Traditional NCA with scaling
    start_time = time.time()
    nca_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('nca', NeighborhoodComponentsAnalysis(n_components=n_components, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])
    nca_pipeline.fit(X_train, y_train)
    nca_time = time.time() - start_time
    nca_pred = nca_pipeline.predict(X_test)

    # DW-NCA with scaling
    start_time = time.time()
    dwnca_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Ensure data is scaled
        ('dwnca_knn', DWNCA_KNNClassifier(n_neighbors=n_neighbors, n_components=n_components))
    ])
    dwnca_pipeline.fit(X_train, y_train)
    dwnca_time = time.time() - start_time
    dwnca_pred = dwnca_pipeline.predict(X_test)

    # Performance metrics
    nca_metrics = PerformanceMetrics(y_test, nca_pred)
    dwnca_metrics = PerformanceMetrics(y_test, dwnca_pred)

    # Cross-validation scores with scaling
    nca_cv_scores = cross_val_score(nca_pipeline, X, y, cv=cv)
    dwnca_cv_scores = cross_val_score(dwnca_pipeline, X, y, cv=cv)

    return {
        'NCA': {
            'time': nca_time,
            'metrics': nca_metrics,
            'cv_scores': nca_cv_scores
        },
        'DW-NCA': {
            'time': dwnca_time,
            'metrics': dwnca_metrics,
            'cv_scores': dwnca_cv_scores
        }
    }

def print_comparison_results(results):
    """
    Print the comparison results for NCA and DW-NCA.

    Parameters:
    ----------
    results : dict
        Dictionary with performance metrics for NCA and DW-NCA.
    """
    for method, data in results.items():
        print(f"\n{method}:")
        print(f"  Time: {data['time']:.4f} seconds")
        print(f"  Accuracy: {data['metrics'].accuracy:.4f}")
        print(f"  Precision: {data['metrics'].precision:.4f}")
        print(f"  Recall: {data['metrics'].recall:.4f}")
        print(f"  F1 Score: {data['metrics'].f1:.4f}")
        print(f"  Cross-validation Score: {np.mean(data['cv_scores']):.4f} ± {np.std(data['cv_scores']):.4f}")

def main():
    """
    Main function to load datasets, compare methods, and print results.
    """
    datasets = load_datasets()
    for name, data in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"Dataset: {name}")
        print(f"{'=' * 50}")
        
        X, y = data.data, data.target
        results = compare_methods(X, y)
        print_comparison_results(results)

if __name__ == "__main__":
    main()
