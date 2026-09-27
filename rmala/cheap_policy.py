"""Fold a min-Hamming preselector into the existing sketch precheck.

Frozen policy/bank, finite gate outputs. No second sketch scan or pre-MLP.
"""
from .utility_gate import UtilityRouter


class CheapSketchRouter(UtilityRouter):
    def __init__(self,bank,post,choice,pre_hamming):
        h=min(choice['hamming'],pre_hamming)
        super().__init__(bank,post,choice['threshold'],h)
        self.closed=pre_hamming<0
        self.pre_evaluations=0;self.extra_sketch_comparisons=0

    def query(self,q,base,allowed=True):
        return super().query(q,base,allowed and not self.closed and self.threshold<=1)
