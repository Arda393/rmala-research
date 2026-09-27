import unittest
import torch
from rmala.pre_gate import pre_features
from rmala.rejection import signature


class PreGateTests(unittest.TestCase):
    def test_only_compact_sketch_and_observables_required(self):
        q=torch.arange(16).float();base=torch.ones(16)
        f=pre_features(q,base,[signature(q)],.5)
        self.assertEqual(f.shape,(8,));self.assertTrue(torch.isfinite(f).all())
        self.assertEqual(float(f[-2]),0.);self.assertEqual(float(f[-1]),.5)

    def test_empty_sketch_and_constant_base_are_finite(self):
        f=pre_features(torch.zeros(16),torch.zeros(16),[],0.)
        self.assertTrue(torch.isfinite(f).all());self.assertEqual(float(f[-2]),1.)
