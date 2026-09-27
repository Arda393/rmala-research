import unittest
import torch
from rmala.budget_memory import PrefixBudget,BudgetMemory,pack,unpack,packed_bytes
from rmala.attention import ResidualAttention


class BudgetTests(unittest.TestCase):
    def test_strict_every_prefix_cap(self):
        for rate in (0,.01,.02,.05,.1,1):
            budget=PrefixBudget(rate)
            for i in range(1,1001):
                budget.advance();budget.consume();budget.consume()
                self.assertLessEqual(budget.used,int(rate*i+1e-9))

    def test_int8_is_real_and_symmetric_for_both_payloads(self):
        value=torch.linspace(-3,3,16)
        encoded=pack(value,True)
        self.assertEqual(encoded[0].dtype,torch.int8)
        self.assertEqual(packed_bytes(encoded),20)
        self.assertLess(float((unpack(encoded)-value).abs().max()),3/127)
        raw=BudgetMemory(16,16,1024,payload='raw',int8=True)
        residual=BudgetMemory(16,16,1024,payload='residual',int8=True)
        self.assertEqual(raw.capacity,residual.capacity)

    def test_candidates_and_anchors_count_against_budget(self):
        ordinary=BudgetMemory(4,4,512,payload='raw')
        enhanced=BudgetMemory(4,4,512,payload='anchored',candidates=2,write_rate=1,tau=0)
        self.assertLess(enhanced.capacity,ordinary.capacity)
        for _ in range(20):
            enhanced.observe(torch.ones(4),torch.ones(4),torch.zeros(4),lambda k:torch.zeros(4))
            self.assertLessEqual(enhanced.tensor_bytes,512)

    def test_anchored_payload_corrects_changed_base_exactly(self):
        memory=BudgetMemory(2,2,1024,payload='anchored',write_rate=1,read_rate=1,tau=0,topk=1)
        key=torch.tensor([1.,0.]);value=torch.tensor([2.,4.]);old=torch.tensor([1.5,2.])
        memory.observe(key,value,old,lambda k:old)
        new=torch.tensor([.5,1.])
        torch.testing.assert_close(memory.query(key,new),value)

    def test_closed_gate_does_not_scan_bank(self):
        memory=BudgetMemory(2,2,1024,write_rate=1,read_rate=.05,tau=0)
        memory.observe(torch.ones(2),torch.ones(2),torch.zeros(2),lambda k:torch.zeros(2))
        for _ in range(100): memory.query(torch.ones(2),torch.zeros(2),alpha=0.)
        self.assertEqual(memory.comparisons,0)
        self.assertEqual(memory.queries,0)

    def test_delayed_candidate_rescues_initially_correct_value(self):
        memory=BudgetMemory(2,2,1024,write_rate=.5,read_rate=1,tau=.1,adaptive=False,candidates=2,delay=1)
        key=torch.tensor([1.,0.]);v=torch.tensor([1.,0.])
        memory.observe(key,v,v,lambda k:v)
        # Source token 2 is reconstructed perfectly, but old token is now lost.
        memory.observe(torch.tensor([0.,1.]),v,v,lambda k:torch.zeros(2))
        self.assertEqual(memory.delayed_writes,1)
        self.assertEqual(memory.write_budget.used,1)

    def test_budget_model_gradient_and_inference_limits(self):
        torch.manual_seed(23)
        layer=ResidualAttention(16,2,budget_options=dict(byte_budget=1024,write_rate=.1,
            read_rate=.05,tau=.0,adaptive=False,candidates=2,delay=2))
        x=torch.randn(1,24,16,requires_grad=True)
        y=layer(x);(y.square().mean()+.1*layer.gate_cost).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(layer.budget_gate.weight.grad.abs().sum()),0)
        layer.eval()
        with torch.no_grad(): layer(x.detach())
        self.assertLessEqual(layer.stats['write_rate'],.1)
        self.assertLessEqual(layer.stats['retrieval_rate'],.05)

    def test_budget_attention_causal(self):
        layer=ResidualAttention(8,1,budget_options=dict(byte_budget=1024,write_rate=.5,
            read_rate=.5,tau=0,adaptive=False,candidates=2)).eval()
        x=torch.randn(1,16,8);y=x.clone();y[:,8:]+=10
        with torch.no_grad():
            torch.testing.assert_close(layer(x)[:,:8],layer(y)[:,:8])


if __name__=='__main__': unittest.main()
