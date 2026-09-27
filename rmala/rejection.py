"""Frozen-bank inference router for the v3 mechanism experiment.

The optional 15-bit centered-sign sketch costs two bytes per entry. It scans
sketches, not full keys; this is O(entries), NOT a constant-time membership oracle.
Rebuild the router after any source-bank mutation. No target labels enter routing.
"""
import torch
from torch.nn import functional as F
from .budget_memory import PrefixBudget, unpack


def signature(q):
    if q.numel() < 15:
        raise ValueError('Sketch requires at least 15 dimensions')
    bits=(q[:15] > q.mean()).tolist()
    return sum(int(bit) << i for i,bit in enumerate(bits))


class RejectionRouter:
    def __init__(self,bank,similarity=.99,hamming=None,read_rate=.05,total_bytes=2048):
        self.bank=bank
        self.similarity=similarity
        self.hamming=hamming
        self.budget=PrefixBudget(read_rate)
        self.sketch=torch.tensor([signature(e['key']) for e in bank.entries],dtype=torch.int16) if hamming is not None else torch.empty(0,dtype=torch.int16)
        self.bytes=bank.tensor_bytes+self.sketch.numel()*2
        if self.bytes>total_bytes: raise ValueError('Bank plus index exceeds budget')
        self.attempts=self.applied=self.comparisons=self.prechecks=self.sketch_comparisons=0
        self.index_build_key_elements=len(bank.entries)*bank.key_dim if hamming is not None else 0

    def query(self,q,base,allowed=True):
        self.budget.advance()
        if not allowed or not self.bank.entries or not self.budget.available:
            return base,False,None
        if self.hamming is not None:
            self.prechecks+=1
            sig=signature(q)
            self.sketch_comparisons+=len(self.sketch)
            if min((sig ^ int(s)).bit_count() for s in self.sketch)>self.hamming:
                return base,False,None
        assert self.budget.consume()
        self.attempts+=1
        self.comparisons+=len(self.bank.entries)
        keys=torch.stack([e['key'] for e in self.bank.entries])
        scores=F.normalize(keys,dim=-1)@F.normalize(q.float(),dim=-1)
        score,idx=scores.max(0)
        if round(float(score),5)<self.similarity: return base,False,None
        entry=self.bank.entries[int(idx)]
        value=unpack(entry['payload'])
        if self.bank.payload=='residual': value=base+value
        elif self.bank.payload=='anchored': value=value+entry['anchor']
        self.applied+=1
        return value,True,int(entry['meta'][0])-1
