import unittest
import numpy as np
import pandas as pd
from topic1.run import features, pipeline, scores


class Topic1ProtocolTests(unittest.TestCase):
    def test_formula_scaling_is_invariant(self):
        matrices, names, _ = features(pd.Series(['Fe2O3','Fe4O6'], index=['a','b']))
        for matrix in matrices.values():
            np.testing.assert_allclose(matrix[0],matrix[1],equal_nan=True)
        np.testing.assert_allclose(matrices['C'],np.column_stack([matrices['A'],matrices['B']]))

    def test_preprocessing_does_not_fit_test_data(self):
        train = np.array([[1,np.nan],[3,10],[5,20],[7,30]])
        p = pipeline('Linear',False).fit(train,[1,2,3,4])
        np.testing.assert_allclose(p[0].statistics_,[4,20])
        before = p[1].mean_.copy()
        p.predict([[1000000,np.nan]])
        np.testing.assert_array_equal(before,p[1].mean_)

    def test_auc_uses_probabilities(self):
        s = scores([0,0,1,1],np.array([.1,.2,.3,.4]),True)
        self.assertEqual(s['roc_auc'],1.0)
        self.assertEqual(s['balanced_accuracy'],.5)

    def test_three_substantive_models_plus_dummy(self):
        x = np.arange(80,dtype=float).reshape(20,4)
        for classification in [False,True]:
            y = np.arange(20)%2 if classification else np.arange(20)
            for model in ['Linear','RF','ET','Dummy']:
                p = pipeline(model,classification).fit(x,y)
                pred = p.predict_proba(x[:2]) if classification else p.predict(x[:2])
                self.assertTrue(np.isfinite(pred).all())


if __name__ == '__main__':
    unittest.main()
