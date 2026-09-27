import unittest
import torch
from rmala.budget_memory import BudgetMemory
from rmala.utility_gate import UtilityGate
from rmala.raw_verifier import RawSketchVerifier
from rmala.rejection import signature


class RawVerifierTests(unittest.TestCase):
    def make(self,h,total=2048):
        bank=BudgetMemory(16,16,payload='raw',write_rate=1,tau=0,adaptive=False)
        q=torch.arange(1,17).float();base=torch.zeros(16);bank.observe(q,torch.ones(16),base,lambda _:base)
        gate=UtilityGate()
        for p in gate.parameters():p.data.zero_()
        r=RawSketchVerifier(bank,gate,dict(threshold=.4,hamming=4),torch.tensor([signature(q)],dtype=torch.int16),h,total)
        r.budget.rate=1
        return r,q,base

    def test_wrong_original_key_rejects_but_consumes_read(self):
        r,q,b=self.make(0)
        with torch.no_grad():v,a,_=r.query(q,b,-q)
        self.assertFalse(a);self.assertTrue(torch.equal(v,b));self.assertEqual(r.applied,0)
        self.assertEqual(r.attempts,1);self.assertEqual(r.verifier_rejections,1)

    def test_matching_key_and_permissive_bypass(self):
        for h in [0,15,None]:
            r,q,b=self.make(h)
            with torch.no_grad():v,a,_=r.query(q,b,q)
            self.assertTrue(a);self.assertTrue(torch.equal(v,torch.ones(16)))
            self.assertEqual(r.bytes,r.bank.tensor_bytes+2+(0 if h is None else 2))

    def test_closed_path_and_missing_query(self):
        r,q,b=self.make(-1)
        with torch.no_grad():self.assertFalse(r.query(q,b)[1])
        self.assertEqual(r.attempts,0)
        r,q,b=self.make(0)
        with self.assertRaises(ValueError):r.query(q,b)

    def test_verifier_bytes_cannot_escape_budget(self):
        with self.assertRaises(ValueError):self.make(0,155)
