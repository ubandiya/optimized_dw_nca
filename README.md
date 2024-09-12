# Optimized Distance-Weighted Neighborhood Component Analysis (DW-NCA)

## Overview

Optimized Distance-Weighted Neighborhood Component Analysis (DW-NCA) is an advanced variant of Neighborhood Component Analysis (NCA) designed to improve the learning of feature transformations by incorporating distance weighting. This method optimizes the separation between classes in a lower-dimensional space while accounting for the distance between data points, enhancing classification performance.

## Mathematical Formulation

### Objective Function

The objective function for DW-NCA is given by:

$`\text{Objective}(A) = \sum_{i \neq j} \text{Weight}_{ij} \cdot \exp(-\text{Distance}_{ij})`$

where:

- $`\text{Weight}_{ij}`$ is the weight based on distance between data points $`i`$ and $`j`$.
- $`\text{Distance}_{ij}`$ is the pairwise squared distance between the transformed data points $`i`$ and $`j`$.

The weights are computed as:

$`\text{Weight}_{ij} = \frac{1}{1 + \text{Distance}_{ij}`$

### Gradient Computation

The gradient of the objective function is used for optimization and is computed as:

$$\text{Gradient} = \frac{\partial}{\partial A} \text{Objective}(A)$$

The gradient is used to update the transformation matrix $`A`$ during optimization.

## Implementation

### DistanceWeightedNCA

The `DistanceWeightedNCA` class in the `optimized_dw_nca` package performs the optimization of the transformation matrix $`A`$ using gradient-based methods. Key parameters include:

- `n_components`: Number of components for the transformation.
- `max_iter`: Maximum number of iterations for the optimization.
- `tol`: Tolerance for optimization convergence.
- `learning_rate`: Learning rate for gradient updates.
- `reg_lambda`: Regularization parameter to prevent overfitting.
- `use_sgd`: Flag to use Stochastic Gradient Descent (SGD) for optimization.

### Example Usage

Here's a basic example of how to use the `DistanceWeightedNCA` class:

```python
from optimized_dw_nca import DistanceWeightedNCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create DistanceWeightedNCA instance
dwnca = DistanceWeightedNCA(n_components=2, learning_rate=1e-2, max_iter=500)

# Fit DW-NCA and transform data
X_train_transformed = dwnca.fit_transform(X_train, y_train)
X_test_transformed = dwnca.transform(X_test)

# Use K-Nearest Neighbors for classification
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_transformed, y_train)
y_pred = knn.predict(X_test_transformed)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Author

This package is developed and maintained by [UBANDIYA Najib Yusuf](https://github.com/ubandiya).

For more information, please visit [my GitHub profile](https://github.com/ubandiya).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
