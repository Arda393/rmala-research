import unittest
import torch
from rmala.utility_gate import UtilityGate,UtilityRouter,utility_loss
from rmala.budget_memory import BudgetMemory


class UtilityTests(unittest.TestCase):
    def test_loss_pushes_harmful_and_useful_oppositely(self):
        logits=torch.zeros(2,requires_grad=True)
        loss=utility_loss(logits,torch.tensor([1.,4.]),torch.tensor([3.,1.]))
        loss.backward()
        self.assertGreater(float(logits.grad[0]),0)
        self.assertLess(float(logits.grad[1]),0)

    def test_rejected_read_still_costs_budget(self):
        with torch.no_grad():
            b=BudgetMemory(16,16,payload='raw',write_rate=1,tau=0,adaptive=False)
            key=torch.arange(1,17).float();base=torch.zeros(16)
            b.observe(key,torch.ones(16),base,lambda q:base)
            gate=UtilityGate()
            for p in gate.parameters():p.zero_()
            gate.net[-1].bias.fill_(-20)
            r=UtilityRouter(b,gate);r.budget.rate=1
            y,a,_=r.query(key,base)
            self.assertFalse(a);self.assertTrue(torch.equal(y,base))
            self.assertEqual(r.attempts,1);self.assertEqual(r.budget.used,1)
            self.assertEqual(r.gate_evaluations,1);self.assertEqual(gate.tensor_bytes,708)
