import unittest
import torch
from rmala.experiments_v4 import safe_errors


class ConstraintTests(unittest.TestCase):
    def context(self):
        return dict(q=torch.zeros(4,16),noisy=torch.tensor([False,False,True,True]),
                    rare=torch.tensor([False,True,False,True]),
                    stored=torch.tensor([False,True,False,True]),base_error=torch.ones(4))

    def test_total_gain_cannot_hide_ordinary_damage(self):
        c=self.context();err=torch.tensor([1.,0.,1.1,0.])
        self.assertLess(float(err.mean()),1.)
        self.assertFalse(safe_errors(c,err))

    def test_regime_damage_cannot_hide_in_pool(self):
        self.assertFalse(safe_errors(self.context(),torch.tensor([.5,1.,1.1,1.])))

    def test_parity_is_feasible_not_quality_success(self):
        self.assertTrue(safe_errors(self.context(),torch.ones(4)))
        self.assertTrue(safe_errors(self.context(),torch.tensor([1.,.5,1.,.5])))
