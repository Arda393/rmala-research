import unittest
from rmala.counterfactual import replay


class CounterfactualTests(unittest.TestCase):
    def test_rejection_spends_credit(self):
        r=replay([1]*4,[0]*4,[0,0,1,1],[0]*4,.5,0,rate=.5)
        self.assertEqual(r['mask'],[False,False,False,True])
        self.assertEqual(r['attempted_reads'],2)

    def test_precheck_preserves_credit(self):
        r=replay([1]*4,[0]*4,[1]*4,[9,9,0,0],.5,0,rate=.5)
        self.assertEqual(r['mask'],[False,False,True,True])

    def test_bypass_can_harm_oracle_cannot(self):
        args=([1,1],[0,3],[0,0],[0,0],.5,0)
        b=replay(*args,rate=1,gate='bypass');o=replay(*args,rate=1,gate='oracle')
        self.assertEqual(b['harmful'],1);self.assertEqual(b['damage_sum'],2)
        self.assertEqual(o['errors'],[0,1]);self.assertEqual(o['attempted_reads'],2)

    def test_empty_bank(self):
        r=replay([1],[0],[1],[0],0,0,rate=1,populated=False)
        self.assertEqual(r['attempted_reads'],0);self.assertEqual(r['errors'],[1])
