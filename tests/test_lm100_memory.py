import unittest
import torch
from rmala.lm100_memory import V14Memory, prefix_accept, historical_threshold
from rmala.budget_memory import BudgetMemory, PrefixBudget
from rmala.utility_gate import UtilityRouter


class CausalMemoryTests(unittest.TestCase):
    def test_prefix_credit_matches_sequential_policy(self):
        torch.manual_seed(13)
        candidates = torch.rand(8, 2048) < .14
        candidates[0] = True
        candidates[1, :1000] = False
        actual, counts = prefix_accept(candidates)
        for row in range(8):
            budget = PrefixBudget(.05)
            expected = []
            for c in candidates[row]:
                budget.advance()
                expected.append(bool(c) and budget.consume())
            self.assertEqual(actual[row].tolist(), expected)
            self.assertTrue((counts[row] <= torch.arange(1,2049)//20).all())

    def test_threshold_only_uses_past(self):
        torch.manual_seed(7)
        error=torch.rand(3,140)
        actual=historical_threshold(error)
        for row in range(3):
            for t in range(140):
                h=sorted(error[row,max(0,t-64):t].tolist())
                expected=.3 if len(h)<8 else max(.3,h[int(.95*len(h))])
                self.assertAlmostEqual(float(actual[row,t]),expected,places=6)
        changed=error.clone()
        changed[:,80:]=100
        self.assertTrue(torch.equal(actual[:,:81],historical_threshold(changed)[:,:81]))

    def example(self):
        torch.manual_seed(34)
        keys=torch.randn(4,16)
        k=keys[torch.arange(120)%4][None,:,None]
        q=keys[(torch.arange(120)+1)%4][None,:,None]
        v=torch.randn(1,120,1,16)
        base=torch.randn_like(v)*.25
        rec=torch.zeros_like(v)
        mem=V14Memory(value_dim=16).eval()
        with torch.no_grad():
            mem.gate.net[-1].weight.zero_()
            mem.gate.net[-1].bias.fill_(8)
        return mem,q,k,v,base,rec

    def test_live_reference_parity_and_real_int8(self):
        mem,q,k,v,base,rec=self.example()
        got=mem(q,k,v,base,rec,diagnostics=True)
        bank=BudgetMemory(16,16,byte_budget=2048-2*mem.capacity,payload='raw',int8=True)
        budget=PrefixBudget(.05)
        expected=[]
        with torch.no_grad():
            for t in range(q.shape[1]):
                bank.observe(k[0,t,0],v[0,t,0],rec[0,t,0],None)
                router=UtilityRouter(bank,mem.gate,threshold=.99,hamming=1)
                router.budget=budget
                y,_,_=router.query(q[0,t,0],base[0,t,0])
                expected.append(y)
        self.assertTrue(torch.allclose(got[0,:,0],torch.stack(expected),atol=1e-6))
        self.assertEqual(int(mem.stats['writes']),bank.write_budget.used)
        self.assertEqual(int(mem.stats['reads']),budget.used)
        self.assertGreater(int(mem.stats['accepted']),0)
        self.assertLessEqual(mem.stats['bank_bytes_per_head'],2048)

    def test_no_future_leakage_and_gate_gradient(self):
        mem,q,k,v,base,rec=self.example()
        before=mem(q,k,v,base,rec).clone()
        k2,v2=k.clone(),v.clone()
        k2[:,81:]=torch.randn_like(k2[:,81:])*100
        v2[:,81:]=torch.randn_like(v2[:,81:])*100
        after=mem(q,k2,v2,base,rec)
        self.assertTrue(torch.equal(before[:,:81],after[:,:81]))
        mem.train()
        with torch.no_grad():
            mem.gate.net[-1].bias.zero_()
        base=base.requires_grad_()
        out=mem(q,k,v,base,rec)
        # Hard forward remains baseline when p=.5 < .99; surrogate still learns.
        self.assertTrue(torch.equal(out,base))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(base.grad).all())
        self.assertGreater(float(mem.gate.net[-1].bias.grad.abs().sum()),0)


if __name__=='__main__':
    unittest.main()
