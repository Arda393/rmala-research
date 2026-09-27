import unittest
import torch
from rmala.budget_memory import BudgetMemory
from rmala.rejection import RejectionRouter


class RejectionTests(unittest.TestCase):
    def bank(self):
        b=BudgetMemory(16,16,payload='raw',write_rate=1,adaptive=False,tau=0)
        k=torch.arange(1,17).float()
        b.observe(k,torch.ones(16),torch.zeros(16),lambda q:torch.zeros(16))
        return b,k

    def test_rejection_cost_and_exact_recall(self):
        b,k=self.bank();r=RejectionRouter(b,similarity=.999,read_rate=1)
        base=torch.zeros(16)
        y,ok,_=r.query(k.flip(0),base)
        self.assertFalse(ok);self.assertTrue(torch.equal(y,base))
        y,ok,idx=r.query(k,base)
        self.assertTrue(ok);self.assertEqual(idx,0)
        self.assertTrue(torch.equal(y,torch.ones(16)))
        self.assertEqual(r.attempts,2);self.assertEqual(r.applied,1)

    def test_precheck_and_prefix(self):
        b,k=self.bank();r=RejectionRouter(b,hamming=0,read_rate=.05)
        for i in range(40):
            r.query(k.flip(0) if i<30 else k,torch.zeros(16))
            self.assertLessEqual(r.attempts,int(.05*(i+1)))
        self.assertGreater(r.prechecks,0)
        self.assertEqual(r.bytes,b.tensor_bytes+2)
        self.assertLess(r.attempts,r.prechecks)

    def test_index_must_fit(self):
        b,k=self.bank()
        with self.assertRaises(ValueError): RejectionRouter(b,hamming=0,total_bytes=b.tensor_bytes)
