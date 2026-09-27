import unittest
import torch
from rmala.hard_replay import replay_batch,weighted_utility_loss
from rmala.utility_gate import utility_loss


class HardReplayTests(unittest.TestCase):
    def test_population_mass_and_unique_sampling(self):
        hard=torch.arange(1000)<10
        ids,w=replay_batch(hard,torch.Generator().manual_seed(2),100,8)
        self.assertEqual(len(ids.unique()),100);self.assertEqual(int(hard[ids].sum()),8)
        self.assertAlmostEqual(float(w[hard[ids]].sum()/100),.01,places=6)
        self.assertAlmostEqual(float(w[~hard[ids]].sum()/100),.99,places=6)

    def test_small_strata_and_empty_fallback(self):
        for count in [0,1,999,1000]:
            h=torch.arange(1000)<count;ids,w=replay_batch(h,torch.Generator().manual_seed(1),100,8)
            self.assertEqual(len(ids.unique()),100);self.assertTrue(torch.isfinite(w).all())
            self.assertAlmostEqual(float(w.mean()),1.,places=5)

    def test_unit_weights_match_loss_and_gradients(self):
        a=torch.tensor([-.5,.5],requires_grad=True);b=a.detach().clone().requires_grad_(True)
        base=torch.tensor([1.,3.]);candidate=torch.tensor([2.,1.])
        x=utility_loss(a,base,candidate);y=weighted_utility_loss(b,base,candidate,torch.ones(2))
        x.backward();y.backward();self.assertEqual(float(x),float(y));self.assertTrue(torch.equal(a.grad,b.grad))

    def test_corrected_two_stratum_loss_matches_population(self):
        # Within-stratum constant losses make a small representative sample exact.
        logits=torch.tensor([-.4]*2+[.8]*98);base=torch.ones(100);candidate=torch.tensor([3.]*2+[0.]*98)
        ids,w=replay_batch(torch.arange(100)<2,torch.Generator().manual_seed(1),10,2)
        a=utility_loss(logits,base,candidate);b=weighted_utility_loss(logits[ids],base[ids],candidate[ids],w)
        self.assertAlmostEqual(float(a),float(b),places=6)
