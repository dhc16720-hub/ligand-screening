import unittest
import numpy as np
from topic1.refine_models import signed_log1p, signed_expm1, make_model, predict, select, combine


class RefinementTests(unittest.TestCase):
    def test_signed_transform_preserves_negative_and_zero(self):
        y = np.array([-1e6, -2., 0., 3., 1e6])
        np.testing.assert_allclose(signed_expm1(signed_log1p(y)), y)

    def test_selection_uses_inner_predictions(self):
        y = np.array([0., 1., 2., 3., 4., 5.])
        inner = [(np.setdiff1d(np.arange(6), b), b) for b in np.array_split(np.arange(6), 3)]
        oofs = dict(good=y+.1, high=y+3, low=y-3)
        recipe, _ = select(y, oofs, inner, False)
        self.assertLess(np.mean(np.abs(combine(oofs, recipe)-y)), .11)
        self.assertNotIn('target', recipe)

    def test_models_predict_finite_full_precision(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(32, 4)); x[0, 0] = np.nan
        for family, params, cl in [('CatBoost', {'depth':4,'loss_function':'MAE'}, False),
                                   ('CatBoost', {'depth':4,'loss_function':'Logloss'}, True),
                                   ('XGB_Log', {'max_depth':3}, False), ('XGB_MAE', {'max_depth':3}, False)]:
            y = np.arange(32)%2 == 0 if cl else rng.normal(size=32)
            model = make_model(family, params, cl).fit(x, y)
            p = predict(model, x, cl)
            self.assertEqual(p.dtype, np.dtype('float64'))
            self.assertTrue(np.isfinite(p).all())
            if cl: self.assertTrue(((p>=0)&(p<=1)).all())


if __name__ == '__main__':
    unittest.main()
