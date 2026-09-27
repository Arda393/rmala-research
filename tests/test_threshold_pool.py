import unittest
from rmala.threshold_audit import candidates


class ThresholdPoolTests(unittest.TestCase):
    def test_empty_bank_scores_cannot_shift_active_grid(self):
        active=[[.001,.002,.003,.004]]
        a=candidates(active,[True],True)
        b=candidates(active+[[.95]*4],[True,False],True)
        self.assertEqual(a,b)
        self.assertNotEqual(a,candidates(active+[[.95]*4],[True,False],False))

    def test_no_populated_bank_has_explicit_constant_options(self):
        self.assertEqual(candidates([[.2]],[False],True),[0.,1.00001])
