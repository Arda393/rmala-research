import unittest
from rmala.v21_threshold_curve import sweep,at_threshold


class ThresholdAuditTests(unittest.TestCase):
    def test_tied_scores_are_atomic_and_harm_is_not_net_loss(self):
        es=[dict(score=.99,delta=-4.,correct=True,missing=False,groups=['a|ordinary']),
            dict(score=.99,delta=2.,correct=False,missing=True,groups=['a|ordinary','a|missing'])]
        rows=sweep(es,{'a|ordinary':2,'a|missing':1},10.,2,2)
        r=at_threshold(rows,.99)
        self.assertEqual(r['accepted'],2);self.assertEqual(r['harm_sum'],2.)
        self.assertEqual(r['net_mse_change'],-1.);self.assertEqual(r['failed_contexts'],1)
        self.assertEqual(at_threshold(rows,.99001)['accepted'],0)
        self.assertEqual(at_threshold(rows,0.)['accepted'],2)

    def test_empty_curve_and_closed_read_cost(self):
        rows=sweep([],{},10.,2,2)
        self.assertEqual(at_threshold(rows,1.00001)['attempted_reads'],0)
        self.assertEqual(at_threshold(rows,.9)['accepted'],0)
        self.assertEqual(at_threshold(rows,.9)['attempted_reads'],2)
