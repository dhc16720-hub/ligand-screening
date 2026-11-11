import unittest
import numpy as np
from experiments.external_diagnostics import metrics, reliability


class ExternalDiagnosticsTests(unittest.TestCase):
    def test_single_class_metrics_are_undefined_not_fabricated(self):
        result=metrics([0,0,0],[.6,.8,.9])
        self.assertIsNone(result['roc_auc'])
        self.assertIsNone(result['average_precision'])
        self.assertIsNone(result['balanced_accuracy'])
        self.assertEqual((result['tn'],result['fp'],result['fn'],result['tp']),(0,3,0,0))

    def test_threshold_tie_matches_frozen_scoring(self):
        result=metrics([0,1],[.5,.5])
        self.assertEqual(result['predicted_positive_fraction'],0)
        self.assertEqual(result['brier'],.25)

    def test_reliability_keeps_endpoints_and_weights_by_sample_count(self):
        bins,ece=reliability([0,1,0],[0,1,.9])
        self.assertEqual(sum(row['n'] for row in bins),3)
        self.assertEqual(bins[9]['n'],2)
        self.assertAlmostEqual(ece,.3)
        self.assertIsNone(bins[1]['observed_fraction'])

    def test_invalid_probabilities_fail(self):
        for predictions in [[-.01,.8],[.2,1.01],[np.nan,.8]]:
            with self.assertRaises(ValueError):metrics([0,1],predictions)
            with self.assertRaises(ValueError):reliability([0,1],predictions)


if __name__=='__main__':unittest.main()
