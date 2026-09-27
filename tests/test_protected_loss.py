import unittest
import torch
from rmala.protected_loss import protected_utility_loss
from rmala.utility_gate import utility_loss


class ProtectedLossTests(unittest.TestCase):
    def test_only_protected_harm_gradient_is_amplified(self):
        base=torch.tensor([1.,1.,3.]);candidate=torch.tensor([3.,3.,1.]);protected=torch.tensor([True,False,True])
        a=torch.zeros(3,requires_grad=True);b=torch.zeros(3,requires_grad=True)
        utility_loss(a,base,candidate).backward()
        protected_utility_loss(b,base,candidate,protected,4.).backward()
        self.assertTrue(torch.allclose(b.grad,a.grad*torch.tensor([4.,1.,1.])))

    def test_multiplier_one_matches_original(self):
        x=torch.tensor([-.5,.5]);b=torch.tensor([1.,3.]);v=torch.tensor([2.,1.])
        self.assertEqual(float(utility_loss(x,b,v)),float(protected_utility_loss(x,b,v,torch.ones(2,dtype=torch.bool),1.)))

    def test_zero_gain_has_zero_finite_loss(self):
        x=torch.zeros(2,requires_grad=True);v=torch.ones(2)
        loss=protected_utility_loss(x,v,v,torch.ones(2,dtype=torch.bool));loss.backward()
        self.assertEqual(float(loss),0.);self.assertTrue(torch.equal(x.grad,torch.zeros(2)))
