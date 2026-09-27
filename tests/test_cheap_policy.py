import unittest
import torch
from rmala.budget_memory import BudgetMemory
from rmala.utility_gate import UtilityGate,UtilityRouter
from rmala.cheap_policy import CheapSketchRouter


class CheapPolicyTests(unittest.TestCase):
    def make(self,h,post_t=.4):
        bank=BudgetMemory(16,16,payload='raw',write_rate=1,tau=0,adaptive=False)
        q=torch.arange(1,17).float();b=torch.zeros(16)
        bank.observe(q,torch.ones(16),b,lambda _:b)
        gate=UtilityGate()
        for p in gate.parameters():p.data.zero_()
        return CheapSketchRouter(bank,gate,dict(threshold=post_t,hamming=4),h),q,b

    def test_closed_selector_never_scans(self):
        r,q,b=self.make(-1)
        with torch.no_grad():
            for _ in range(40):self.assertTrue(torch.equal(r.query(q,b)[0],b))
        self.assertEqual(r.attempts,0);self.assertEqual(r.sketch_comparisons,0)
        self.assertEqual(r.budget.opportunities,40)

    def test_exact_match_single_scan_and_prefix_budget(self):
        r,q,b=self.make(0)
        with torch.no_grad():
            for i in range(40):
                r.query(q,b);self.assertLessEqual(r.attempts,int(.05*(i+1)+1e-9))
        self.assertEqual(r.attempts,2);self.assertEqual(r.sketch_comparisons,2)
        self.assertEqual(r.extra_sketch_comparisons,0);self.assertEqual(r.pre_evaluations,0)

    def test_precheck_rejection_does_not_consume_credit(self):
        r,q,b=self.make(0);r.budget.rate=1
        with torch.no_grad():r.query(-q,b)
        self.assertEqual(r.attempts,0);self.assertEqual(r.budget.used,0)

    def test_constant_closed_post_never_scans(self):
        r,q,b=self.make(4,1.00001);r.budget.rate=1
        with torch.no_grad():r.query(q,b)
        self.assertEqual(r.attempts,0);self.assertEqual(r.sketch_comparisons,0)
