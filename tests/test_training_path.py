import unittest
from types import SimpleNamespace
import torch
from rmala.experiments_v14 import train_path


class TrainingPathTests(unittest.TestCase):
    def test_intermediate_snapshot_does_not_change_final_training(self):
        c=dict(bank=SimpleNamespace(entries=[1]),utility_features=torch.arange(80).float().reshape(10,8)/80,
               base_error=torch.ones(10),out_error=torch.tensor([.5,2.]*5))
        a,_=train_path([c],7,(2,5));b,_=train_path([c],7,(5,))
        for k,v in a[5].state_dict().items():self.assertTrue(torch.equal(v,b[5].state_dict()[k]))
        self.assertTrue(any(not torch.equal(v,a[5].state_dict()[k]) for k,v in a[2].state_dict().items()))

    def test_invalid_step_is_rejected(self):
        with self.assertRaises(ValueError):train_path([],7,(0,))
