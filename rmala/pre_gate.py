"""Pre-retrieval features use only query, base output and compact sketches."""
import torch
from .rejection import signature
from .utility_gate import UtilityGate,UtilityRouter


def pre_features(q,base,sketch,fill):
    distances=[(signature(q)^int(s)).bit_count() for s in sketch]
    d=min(distances) if distances else 15
    return torch.stack([q.abs().max(),q.mean(),base.square().mean().sqrt().log1p(),
        base.abs().max().log1p(),base.mean(),base.std(unbiased=False),
        base.new_tensor(d/15),base.new_tensor(fill)])


class PreRouter(UtilityRouter):
    def __init__(self,bank,post,choice,pre,threshold):
        super().__init__(bank,post,choice['threshold'],choice['hamming'])
        self.pre=pre;self.pre_threshold=threshold;self.pre_evaluations=0;self.extra_sketch_comparisons=0

    def query(self,q,base,allowed=True):
        if allowed and self.bank.entries:
            # Deliberately count every pre-evaluation, including queries without
            # read credit. Parent performs its own precheck; duplication is counted.
            self.pre_evaluations+=1;self.extra_sketch_comparisons+=len(self.sketch)
            f=pre_features(q,base,self.sketch,len(self.bank.entries)/max(1,self.bank.capacity))
            allowed=round(float(self.pre(f).sigmoid()),5)>=self.pre_threshold
        return super().query(q,base,allowed)
