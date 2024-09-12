import unittest
from optimized_dw_nca import PerformanceMetrics

class TestPerformanceMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = [0, 1, 1, 0]
        self.y_pred = [0, 1, 0, 1]
        self.metrics = PerformanceMetrics(self.y_true, self.y_pred)

    def test_accuracy(self):
        accuracy = self.metrics.accuracy
        self.assertAlmostEqual(accuracy, 0.5)

    def test_precision(self):
        precision = self.metrics.precision
        self.assertAlmostEqual(precision, 0.5)

    def test_recall(self):
        recall = self.metrics.recall
        self.assertAlmostEqual(recall, 0.5)

    def test_f1(self):
        f1 = self.metrics.f1
        self.assertAlmostEqual(f1, 0.5)

if __name__ == '__main__':
    unittest.main()
