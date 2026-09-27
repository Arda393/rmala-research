import math
import unittest
import torch
from rmala.attention import ResidualAttention,gla_reference,gla_fla
from rmala.bank import ExactBank
from rmala.model import LanguageModel
from rmala.data import metrics
from rmala.synthetic import batch


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(2)

    def test_gated_scan_against_explicit_causal_sum(self):
        q,k=torch.rand(2,6,2,4),torch.rand(2,6,2,4)
        v=torch.randn(2,6,2,3);g=-torch.rand_like(k)
        out,recon,den=gla_reference(q,k,v,g)
        for t in range(6):
            weights=[]
            for j in range(t+1):
                decay=g[:,j+1:t+1].sum(1).exp()
                weights.append((q[:,t]*decay*k[:,j]).sum(-1))
            w=torch.stack(weights,1)
            expected=(w.unsqueeze(-1)*v[:,:t+1]).sum(1)/w.sum(1).unsqueeze(-1)
            torch.testing.assert_close(out[:,t],expected)
        torch.testing.assert_close(recon[:,0],v[:,0])

    def test_lru_access_protects_recently_read_entry(self):
        bank=ExactBank(2)
        bank.write(torch.tensor([1.,0.,0.]),torch.tensor([10.]))
        bank.write(torch.tensor([0.,1.,0.]),torch.tensor([20.]))
        bank.query(torch.tensor([1.,0.,0.]),1)
        bank.write(torch.tensor([0.,0.,1.]),torch.tensor([30.]))
        self.assertEqual([float(e[1]) for e in bank.entries],[10.,30.])

    def test_capacity_zero_and_payload_byte_equality(self):
        bank=ExactBank(0);bank.write(torch.ones(4),torch.zeros(4))
        self.assertEqual(bank.size,0)
        a,b=ExactBank(1),ExactBank(1)
        a.write(torch.ones(4),torch.zeros(4));b.write(torch.ones(4),torch.randn(4))
        self.assertEqual(a.tensor_bytes,b.tensor_bytes)

    def test_no_future_leakage_and_batch_isolation(self):
        for variant in ("full","gla","rmala","surprise"):
            m=LanguageModel(vocab_size=64,dim=16,heads=2,layers=1,variant=variant,
                            dropout=0,capacity=4,tau=.01).eval()
            x=torch.randint(0,64,(2,8));changed=x.clone();changed[:,5:]=7
            with torch.no_grad():
                a,b=m(x,hard=False),m(changed,hard=False)
                torch.testing.assert_close(a[:,:5],b[:,:5])
                torch.testing.assert_close(a[:1],m(x[:1],hard=False),atol=1e-5,rtol=1e-5)

    def test_bank_autograd_survives_eviction(self):
        m=ResidualAttention(16,2,tau=.01,capacity=2)
        x=torch.randn(1,8,16,requires_grad=True)
        m(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(m.read_gate.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.qkv.weight.grad.abs().sum()),0)

    def test_hard_gate_actually_skips_query(self):
        m=ResidualAttention(16,2,tau=.01,capacity=4).eval()
        with torch.no_grad():
            m.read_gate.weight.zero_();m.read_gate.bias.fill_(-100)
            m(torch.randn(1,8,16),hard=True)
        self.assertGreater(m.stats["write_rate"],0)
        self.assertEqual(m.stats["retrieval_rate"],0)
        self.assertEqual(m.stats["comparisons"],0)

    def test_zero_gate_preserves_base_additively(self):
        m=ResidualAttention(16,2,tau=.01,capacity=4).eval()
        base=ResidualAttention(16,2,variant="gla").eval()
        base.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith("read_gate")})
        with torch.no_grad():
            m.read_gate.weight.zero_();m.read_gate.bias.fill_(-100)
            x=torch.randn(1,8,16)
            torch.testing.assert_close(m(x),base(x))

    def test_metrics_and_synthetic_answer_is_not_input(self):
        m=metrics(10*math.log(2),10,5)
        self.assertAlmostEqual(m["bpb"],2)
        self.assertAlmostEqual(m["perplexity"],2)
        for task in ("niah","multi_needle","mqar"):
            x,y=batch(task,2,32,123)
            self.assertTrue((x[y!=-100]!=y[y!=-100]).all())

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_fla_matches_reference_when_installed(self):
        try:
            from fla.ops.gla import chunk_gla
        except ImportError:
            self.skipTest("fla unavailable")
        q,k=torch.rand(1,32,2,16,device="cuda"),torch.rand(1,32,2,16,device="cuda")
        v=torch.randn_like(q);g=-torch.rand_like(q)/16
        reference=gla_reference(q,k,v,g)
        actual=gla_fla(q,k,v,g)
        for a,b in zip(actual,reference):
            torch.testing.assert_close(a,b,atol=3e-3,rtol=3e-3)


if __name__=="__main__":
    unittest.main()
