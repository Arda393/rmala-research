import unittest
import torch
from rmala.v22_blend import blend


class V22BlendTests(unittest.TestCase):
    def test_endpoints_exact(self):
        b=torch.tensor([1.,2.]);m=torch.tensor([4.,-3.])
        self.assertIs(blend(b,m,0.),b);self.assertIs(blend(b,m,1.),m)
        self.assertTrue(torch.equal(blend(b,m,.5),torch.tensor([2.5,-.5])))

    def test_partial_can_help_when_full_hurts(self):
        b=torch.tensor([0.]);m=torch.tensor([4.]);target=torch.tensor([1.])
        self.assertEqual(float((blend(b,m,.25)-target).square()),0.)
        self.assertGreater(float((blend(b,m,1.)-target).square()),float((b-target).square()))

    def test_smaller_alpha_does_not_guarantee_no_harm(self):
        b=torch.tensor([0.]);m=torch.tensor([4.])
        self.assertGreater(float(blend(b,m,.1).square()),0.)
        self.assertTrue(torch.equal(blend(b,b,.1),b))
        for a in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):blend(b,m,a)
