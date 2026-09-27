import unittest
import torch
from rmala.budget_memory import BudgetMemory
from rmala.utility_gate import UtilityGate
from rmala.pre_gate import PreRouter
from rmala.fast_policy import FrozenFastRouter


class FastPolicyTests(unittest.TestCase):
    def setup_pair(self,pre_t,post_t):
        bank=BudgetMemory(16,16,payload='raw',write_rate=1,tau=0,adaptive=False)
        q=torch.arange(1,17).float();base=torch.zeros(16)
        bank.observe(q,torch.ones(16),base,lambda _:base)
        pre=UtilityGate();post=UtilityGate()
        for model in [pre,post]:
            for p in model.parameters():p.data.zero_()
        args=(bank,post,dict(threshold=post_t,hamming=1),pre,pre_t)
        return PreRouter(*args),FrozenFastRouter(*args),q,base

    def compare(self,pre_t,post_t):
        old,new,q,b=self.setup_pair(pre_t,post_t)
        with torch.no_grad():
            for i in range(40):
                a=old.query(q,b);c=new.query(q,b)
                self.assertTrue(torch.equal(a[0],c[0]));self.assertEqual(a[1:],c[1:])
                self.assertLessEqual(new.attempts,int(.05*(i+1)+1e-9))
        self.assertEqual(new.budget.opportunities,40)
        return old,new

    def test_open_pre_skips_only_pre_work(self):
        old,new=self.compare(0.,.4)
        self.assertEqual(old.attempts,2);self.assertEqual(new.attempts,2)
        self.assertEqual(old.pre_evaluations,40);self.assertEqual(new.pre_evaluations,0)
        self.assertEqual(new.gate_evaluations,2)

    def test_closed_pre_skips_all_work(self):
        old,new=self.compare(1.00001,.4)
        self.assertEqual(old.attempts,0);self.assertEqual(new.attempts,0)
        self.assertEqual(new.pre_evaluations,0);self.assertEqual(new.extra_sketch_comparisons,0)

    def test_closed_post_preserves_output_without_spending_credit(self):
        old,new=self.compare(.4,1.00001)
        self.assertEqual(old.attempts,2);self.assertEqual(new.attempts,0)
        self.assertEqual(new.pre_evaluations,0);self.assertEqual(new.gate_evaluations,0)

    def test_nonconstant_policy_keeps_reference_behavior(self):
        old,new=self.compare(.4,.4)
        self.assertEqual(old.pre_evaluations,new.pre_evaluations)
        self.assertEqual(old.attempts,new.attempts)

    def test_threshold_one_is_not_assumed_closed(self):
        _,new,q,b=self.setup_pair(1.,1.)
        new.pre.net[-1].bias.data.fill_(100);new.gate.net[-1].bias.data.fill_(100)
        new.budget.rate=1
        with torch.no_grad():_,applied,_=new.query(q,b)
        self.assertTrue(applied);self.assertEqual(new.attempts,1)

    def test_threshold_mutation_requires_new_router(self):
        _,new,q,b=self.setup_pair(.4,.4);new.threshold=1.00001
        with self.assertRaises(RuntimeError):new.query(q,b)
