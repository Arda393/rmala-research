"""Constant-policy fast paths, separate from archived V8/V10 reference routers.

Frozen banks, weights and thresholds with finite gate outputs are required.
Closed post-gate streams retain outputs but intentionally spend no read credit.
"""
from .pre_gate import PreRouter
from .utility_gate import UtilityRouter


class FrozenFastRouter(PreRouter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._fixed_thresholds=(self.pre_threshold,self.threshold)

    def query(self,q,base,allowed=True):
        if self._fixed_thresholds!=(self.pre_threshold,self.threshold):
            raise RuntimeError('Rebuild FrozenFastRouter when changing policy thresholds')
        if self.pre_threshold>1 or self.threshold>1:
            self.budget.advance()
            return base,False,None
        if self.pre_threshold<=0:
            return UtilityRouter.query(self,q,base,allowed)
        return super().query(q,base,allowed)
