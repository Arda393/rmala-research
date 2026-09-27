"""Match supervision is training-only; inference sees the same eight features."""
import torch
from torch import nn


def match_targets(ids,selected_ids):
    if ids.shape!=selected_ids.shape:raise ValueError('Mismatched identities')
    return (ids>=0)&(ids==selected_ids)


class GuardedGate(nn.Module):
    def __init__(self,base,guard,base_threshold,guard_threshold):
        super().__init__();self.base=base;self.guard=guard
        self.base_threshold=base_threshold;self.guard_threshold=guard_threshold
        self.base_evaluations=self.guard_evaluations=0

    @property
    def tensor_bytes(self):return self.base.tensor_bytes+self.guard.tensor_bytes

    def forward(self,x):
        if x.ndim!=1:raise ValueError('Streaming single-query inference required')
        self.base_evaluations+=1
        accept=round(float(self.base(x).sigmoid()),5)>=self.base_threshold
        if accept and self.guard_threshold>0:
            self.guard_evaluations+=1
            accept=round(float(self.guard(x).sigmoid()),5)>=self.guard_threshold
        # Outer router uses threshold .5. Original scores retain Python rounding.
        return x.new_tensor(20. if accept else -20.)
