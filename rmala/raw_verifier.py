"""Frozen-bank candidate verification using an original-key 15-bit sketch."""
import torch
from .cheap_policy import CheapSketchRouter
from .rejection import signature


class RawSketchVerifier(CheapSketchRouter):
    def __init__(self,bank,gate,choice,raw_sketch=None,raw_hamming=None,total_bytes=2048):
        super().__init__(bank,gate,choice,choice['hamming'])
        if raw_hamming is not None and (not isinstance(raw_hamming,int) or not -1<=raw_hamming<=15):
            raise ValueError('Invalid original-key Hamming threshold')
        self.raw_hamming=raw_hamming
        self.raw_sketch=torch.empty(0,dtype=torch.int16)
        if raw_hamming is not None:
            if raw_sketch is None or raw_sketch.dtype!=torch.int16 or raw_sketch.ndim!=1 or len(raw_sketch)!=len(bank.entries):
                raise ValueError('One int16 original-key sketch required per entry')
            self.raw_sketch=raw_sketch.clone()
        self.bytes+=self.raw_sketch.numel()*self.raw_sketch.element_size()
        if self.bytes>total_bytes:raise ValueError('Original-key verifier exceeds total bank budget')
        self.verifier_checks=self.verifier_rejections=self.metadata_checks=self.raw_signature_elements=0

    def query(self,q,base,raw_query=None,allowed=True):
        if self.raw_hamming==-1:return super().query(q,base,False)
        if self.raw_hamming is not None and raw_query is None:raise ValueError('Original observable query required')
        value,applied,identity=super().query(q,base,allowed)
        if not applied or self.raw_hamming is None:return value,applied,identity
        # Existing stored metadata locates the chosen slot; no query identity label.
        index=None
        for j,entry in enumerate(self.bank.entries):
            self.metadata_checks+=1
            if int(entry['meta'][0])-1==identity:index=j;break
        assert index is not None
        self.verifier_checks+=1;self.raw_signature_elements+=raw_query.numel()
        distance=(signature(raw_query)^int(self.raw_sketch[index])).bit_count()
        if distance>self.raw_hamming:
            self.verifier_rejections+=1;self.applied-=1
            return base,False,None
        return value,True,identity
