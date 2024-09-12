import unittest
import numpy as np
from optimized_dw_nca import DistanceWeightedNCA, DWNCA_KNNClassifier

class TestDistanceWeightedNCA(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6]])
        self.y = np.array([0, 1, 0])
        self.dwnca = DistanceWeightedNCA(n_components=2)

    def test_fit(self):
        self.dwnca.fit(self.X, self.y)
        self.assertTrue(hasattr(self.dwnca, 'components_'))

    def test_transform(self):
        self.dwnca.fit(self.X, self.y)
        transformed = self.dwnca.transform(self.X)
        self.assertEqual(transformed.shape[1], 2)

    def test_dwnca_knn_classifier(self):
        clf = DWNCA_KNNClassifier(n_neighbors=3)
        clf.fit(self.X, self.y)
        predictions = clf.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

if __name__ == '__main__':
    unittest.main()
