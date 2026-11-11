import unittest
import warnings
import numpy as np
import pandas as pd
from topic1.physics_features import alloy_features,structure_features
from topic1.physics_models import gate


class PhysicsTests(unittest.TestCase):
    def test_local_geometry_translation_invariance(self):
        from pymatgen.core import Lattice,Structure
        s=Structure(Lattice.cubic(3),['Fe'],[[0,0,0]])
        moved=s.copy();moved.translate_sites([0],[.2,.3,.4],frac_coords=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x,names,issues=structure_features(pd.Series([s,moved]),n_jobs=1)
        self.assertFalse(issues)
        self.assertEqual(x.shape[1],len(names))
        np.testing.assert_allclose(x[0],x[1],atol=1e-8)

    def test_alloy_scale_invariance_and_missing_data_retained(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x,labels,issues=alloy_features(pd.Series(['FeNi','Fe2Ni2','He']))
        np.testing.assert_allclose(x[0],x[1],equal_nan=True)
        self.assertEqual(x.shape,(3,5))
        self.assertEqual(len(labels),5)
        self.assertTrue(np.isnan(x[2]).any())
        self.assertTrue(issues)

    def test_gate_accepts_consistent_gain(self):
        y=np.arange(9,dtype=float)
        inner=[(np.setdiff1d(np.arange(9),b),b) for b in np.array_split(np.arange(9),3)]
        self.assertTrue(gate(y,y+2,y+1,inner,False)['use_new'])
        self.assertFalse(gate(y,y+1,y+1,inner,False)['use_new'])

    def test_gate_rejects_gain_concentrated_in_one_fold(self):
        y=np.zeros(9);old=np.ones(9)*2;new=np.array([0.,0.,0.,2.1,2.1,2.1,2.1,2.1,2.1])
        inner=[(np.setdiff1d(np.arange(9),b),b) for b in np.array_split(np.arange(9),3)]
        result=gate(y,old,new,inner,False)
        self.assertGreater(result['mean_gain'],0)
        self.assertFalse(result['use_new'])


if __name__=='__main__':unittest.main()
