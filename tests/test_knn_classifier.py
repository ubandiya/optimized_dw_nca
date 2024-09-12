import unittest
import numpy as np
from optimized_dw_nca import KNNClassifier

class TestKNNClassifier(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6]])
        self.y = np.array([0, 1, 0])
        self.knn = KNNClassifier(n_neighbors=2)

    def test_fit(self):
        self.knn.fit(self.X, self.y)
        self.assertTrue(hasattr(self.knn, 'classifier_'))

    def test_predict(self):
        self.knn.fit(self.X, self.y)
        predictions = self.knn.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

if __name__ == '__main__':
    unittest.main()
