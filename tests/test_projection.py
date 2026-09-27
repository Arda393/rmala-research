import copy
import unittest
import torch
from torch import nn
from rmala.experiments_v4 import mixed
from rmala.experiments_v5 import attributes,projected


class ProjectionTests(unittest.TestCase):
    def test_context_is_shared_not_unique_identity(self):
        cb,ns=attributes(610)
        self.assertEqual(cb.shape,(32,16))
        self.assertLessEqual(len(ns.unique()),32)
        self.assertEqual(len(ns),512)

    def test_projection_does_not_use_targets_or_membership(self):
        with torch.no_grad():
            a=mixed('dense_random',610,'raw_int8');b=copy.deepcopy(a)
            b['y']=torch.randn_like(b['y']);b['stored']=~b['stored']
            torch.manual_seed(3);model=nn.Linear(32,16,bias=False)
            a=projected(a,'dense_random',610,'learned_context',model)
            b=projected(b,'dense_random',610,'learned_context',model)
            self.assertTrue(torch.equal(a['q'],b['q']))
            self.assertEqual(a['score'],b['score'])
            self.assertEqual(a['distance'],b['distance'])
            self.assertLessEqual(a['bank'].tensor_bytes+2*len(a['bank'].entries),2048)
