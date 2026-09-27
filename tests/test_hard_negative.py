import unittest
import torch
from rmala.experiments_v6 import negative_views,operation_counts


class HardNegativeTests(unittest.TestCase):
    def test_collision_and_noncollision_examples_present(self):
        rng=torch.Generator().manual_seed(22)
        k=torch.ones(128,16);c=torch.ones(128,16)*4
        nk,nc,collision=negative_views(k,c,rng)
        self.assertTrue(collision.any());self.assertTrue((~collision).any())
        self.assertLess(float((nc[collision]-c[collision]).abs().mean()),.1)
        self.assertGreater(float((nc[~collision]-c[~collision]).abs().mean()),1.)
        self.assertFalse(torch.equal(nk,k))

    def test_operation_accounting_includes_projection(self):
        row=dict(stored_count=36,queries=1280,attempted_reads=64,prechecks=600,sketch_comparisons=10800)
        c=operation_counts(row,True)
        self.assertEqual(c['query_projection_arithmetic_ops'],1280*1008)
        self.assertEqual(c['router_score_arithmetic_ops'],64*(18*79+48))
        self.assertLess(c['modeled_query_saving_percent'],95.)
        self.assertEqual(c['projection_parameter_bytes'],2048)
