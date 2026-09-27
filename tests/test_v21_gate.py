import unittest
import torch
from rmala.v21_gate import label_pools,fine_cuts,train
from rmala.utility_gate import UtilityGate


class V21Tests(unittest.TestCase):
    def context(self):
        x=torch.zeros(4,8);x[:,0]=torch.tensor([.999,.999,.2,.999])
        return dict(ids=torch.tensor([-1,2,3,4]),selected_ids=torch.tensor([1,1,3,4]),stored=torch.tensor([False,True,True,True]),
            distance=[0,0,3,0],utility_features=x,base_error=torch.ones(4),out_error=torch.tensor([2.,2.,.5,2.]))

    def test_missing_and_wrong_present_are_distinct(self):
        missing,hard,pos=label_pools(self.context())
        self.assertEqual(missing.tolist(),[True,False,False,False])
        self.assertEqual(hard.tolist(),[True,True,False,False])
        self.assertEqual(pos.tolist(),[False,False,True,False])

    def test_high_confidence_search_and_closed_fallback(self):
        cuts=fine_cuts([.99937,.99991])
        self.assertIn(.99937,cuts);self.assertIn(.9999,cuts);self.assertIn(1.00001,cuts)
        self.assertEqual(cuts,sorted(set(cuts)))

    def test_continuation_keeps_base_and_normalization(self):
        base=UtilityGate();old={k:v.clone() for k,v in base.state_dict().items()}
        g,info=train([self.context()],base,7,True,steps=2)
        self.assertTrue(all(torch.equal(v,old[k]) for k,v in base.state_dict().items()))
        self.assertTrue(torch.equal(g.mean,base.mean));self.assertTrue(torch.equal(g.std,base.std))
        self.assertEqual(g.tensor_bytes,708);self.assertEqual(info['stratum_draws'],[64,64,64])
