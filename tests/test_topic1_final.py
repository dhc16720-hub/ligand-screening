import unittest
import numpy as np
from topic1.final_external import composition_key
from topic1.final_models import FinalBundle
from topic1.refine_models import make_model


class FinalTests(unittest.TestCase):
    def test_overlap_key_scale_and_order_invariance(self):
        self.assertEqual(composition_key('FeNi'),composition_key('Ni2Fe2'))
        self.assertNotEqual(composition_key('FeNi'),composition_key('Fe2Ni'))

    def test_bundle_median_prediction(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.dummy import DummyRegressor
        x=np.ones((5,2));models={}
        for key,value in [('C:a',1.),('C:b',2.),('D:c',100.)]:
            models[key]=make_pipeline(DummyRegressor(strategy='constant',constant=value)).fit(x,np.ones(5))
        b=FinalBundle(models,dict(members=list(models),operation='median'),False)
        np.testing.assert_array_equal(b.predict({'C':x,'D':x}),np.ones(5)*2)


if __name__=='__main__':unittest.main()
