import unittest
import numpy as np
from topic1.advanced import make_model,predict,inner_folds,select_blend,CONFIGS
from topic1.conformal import conformal_radius


class AdvancedProtocolTests(unittest.TestCase):
    def test_conformal_finite_sample_quantile(self):
        self.assertEqual(conformal_radius(np.arange(1,11),.1),10)
        self.assertTrue(np.isinf(conformal_radius([1,2,3],.1)))

    def test_grouped_inner_validation(self):
        groups=np.repeat(np.arange(30),2)
        y=np.tile([0,0,1,1],15)
        splits,grouped=inner_folds(y,groups,True,0)
        self.assertTrue(grouped)
        validation=[]
        for a,b in splits:
            self.assertFalse(set(groups[a]) & set(groups[b]))
            validation+=list(b)
        self.assertEqual(sorted(validation),list(range(60)))

    def test_robust_scaling_handles_rare_small_scale_feature(self):
        x=np.column_stack([np.arange(30),np.r_[np.zeros(29),2e-5]])
        model=make_model('Linear',{'regularization':10.},False).fit(x,np.arange(30)*.1)
        self.assertLess(abs(model.predict([[10,1.55]])[0]),100)
        before=model[2].center_.copy()
        model.predict([[1e6,1e6]])
        np.testing.assert_array_equal(model[2].center_,before)

    def test_family_and_blend_selection_is_inner_only(self):
        y=np.arange(12,dtype=float)
        splits=[(np.setdiff1d(np.arange(12),b),b) for b in np.array_split(np.arange(12),3)]
        selected,_=select_blend(y,{'Linear':y+1,'ET':y-1,'RF':y+9,'XGB':y+10},splits,False)
        self.assertEqual(set(selected['members']),{'Linear','ET'})
        self.assertEqual(selected['inner_loss'],0)

    def test_all_models_regression_and_classification(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(60,5))
        for cl in [False,True]:
            y=(x[:,0]>0).astype(int) if cl else x[:,0]*3
            for family in CONFIGS:
                model=make_model(family,CONFIGS[family][0],cl).fit(x,y)
                p=predict(model,x[:3],cl)
                self.assertTrue(np.isfinite(p).all())
                if cl:self.assertTrue(((p>=0)&(p<=1)).all())


if __name__=='__main__':unittest.main()
