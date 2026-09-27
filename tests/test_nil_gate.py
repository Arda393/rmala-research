import unittest
import torch
from rmala.nil_gate import match_targets,GuardedGate
from rmala.utility_gate import UtilityGate


class NILGateTests(unittest.TestCase):
    def gate(self):
        g=UtilityGate()
        for p in g.parameters():p.data.zero_()
        return g

    def test_nil_and_wrong_candidates_are_negative(self):
        y=match_targets(torch.tensor([-1,-1,4,5]),torch.tensor([-1,3,4,6]))
        self.assertEqual(y.tolist(),[False,False,True,False])

    def test_base_rejection_skips_guard(self):
        g=GuardedGate(self.gate(),self.gate(),.6,.4)
        with torch.no_grad():self.assertLess(float(g(torch.zeros(8))),0)
        self.assertEqual(g.base_evaluations,1);self.assertEqual(g.guard_evaluations,0)

    def test_guard_can_only_remove_acceptance(self):
        g=GuardedGate(self.gate(),self.gate(),.4,.6)
        with torch.no_grad():self.assertLess(float(g(torch.zeros(8))),0)
        self.assertEqual(g.guard_evaluations,1);self.assertEqual(g.tensor_bytes,1416)

    def test_disabled_guard_preserves_base_without_guard_compute(self):
        g=GuardedGate(self.gate(),self.gate(),.5,0.)
        with torch.no_grad():self.assertGreater(float(g(torch.zeros(8))),0)
        self.assertEqual(g.guard_evaluations,0)
