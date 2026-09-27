import unittest
import torch
from types import SimpleNamespace
from rmala.experiments_v8 import attribution


class AttributionTests(unittest.TestCase):
    def test_exclusive_stages_follow_actual_budget_order(self):
        c=dict(bank=SimpleNamespace(entries=[1]),distance=[0]*19+[3,0,0],
               stored=torch.ones(22,dtype=torch.bool),base_error=torch.ones(22),out_error=torch.zeros(22))
        detail,mask=attribution(c,[0.]*22,.5,1)
        self.assertEqual(detail['stage_counts'],dict(empty_bank=0,budget=20,precheck=1,post_reject=1,applied=0))
        self.assertEqual(detail['useful_stored_total'],22)
        self.assertEqual(detail['attributed_attempts'],1)
        self.assertFalse(mask.any())

    def test_labels_change_diagnostics_not_decisions(self):
        c=dict(bank=SimpleNamespace(entries=[1]),distance=[0]*40,stored=torch.ones(40,dtype=torch.bool),base_error=torch.ones(40),out_error=torch.zeros(40))
        a,mask=attribution(c,[1.]*40,.5,1)
        c['stored']=~c['stored'];c['base_error'].zero_()
        b,other=attribution(c,[1.]*40,.5,1)
        self.assertTrue(torch.equal(mask,other));self.assertNotEqual(a['useful_stored_total'],b['useful_stored_total'])
