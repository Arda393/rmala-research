import unittest
import torch
from rmala.v23_alpha import scalar_optimum,tensor_blend,train,predict


class V23AlphaTests(unittest.TestCase):
    def test_scalar_optimum_and_bounds(self):
        b=torch.zeros(2,1);m=torch.ones(2,1)*4
        self.assertEqual(scalar_optimum(b,m,torch.ones(2,1)),.25)
        self.assertEqual(scalar_optimum(b,m,-torch.ones(2,1)),0.)
        self.assertEqual(scalar_optimum(b,m,torch.ones(2,1)*8),1.)
        self.assertEqual(scalar_optimum(b,b,b),.5)

    def test_training_starts_half_and_inference_needs_no_labels(self):
        c=dict(utility_features=torch.randn(8,8),base=torch.zeros(8,2),output=torch.ones(8,2)*4,y=torch.ones(8,2))
        with torch.no_grad():h,s,info=train([dict(c=c,eligible=torch.ones(8,dtype=torch.bool))],1,steps=0)
        mask=torch.tensor([True,False]*4)
        a=predict({'utility_features':c['utility_features']},mask,'query',h,s)
        self.assertTrue(torch.equal(a[mask],torch.full((4,),.5)));self.assertEqual(float(a[~mask].sum()),0.)
        self.assertEqual(h.tensor_bytes,708)

    def test_different_alpha_per_query(self):
        y=tensor_blend(torch.zeros(3,1),torch.ones(3,1)*4,torch.tensor([0.,.25,1.]))
        self.assertTrue(torch.equal(y[:,0],torch.tensor([0.,1.,4.])))

    def test_direct_mse_training_improves_over_initial_half(self):
        c=dict(utility_features=torch.randn(8,8),base=torch.zeros(8,2),output=torch.ones(8,2)*4,y=torch.ones(8,2))
        with torch.no_grad():
            h,s,info=train([dict(c=c,eligible=torch.ones(8,dtype=torch.bool))],3,steps=30)
            a=h(c['utility_features']).sigmoid()
            loss=(tensor_blend(c['base'],c['output'],a)-c['y']).square().mean()
        self.assertLess(float(loss),1.)
