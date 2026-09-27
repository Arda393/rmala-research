"""Post-retrieval utility gate: labels are used in training only."""
import torch
from torch import nn
from torch.nn import functional as F
from .rejection import RejectionRouter,signature
from .budget_memory import unpack


def features(scores,base,candidate,fill):
    """Batched or single-row observable features, no target or membership input."""
    if scores.ndim==1:
        return features(scores[None],base[None],candidate[None],fill)[0]
    top=scores.topk(min(2,scores.shape[-1]),dim=-1).values
    first=top[:,0];second=top[:,1] if top.shape[1]>1 else torch.full_like(first,-1)
    # Quantized confidence features reduce batched/single floating-point drift.
    first=(first*1e5).round()/1e5;gap=((top[:,0]-second)*1e5).round()/1e5
    rms=lambda x:x.square().mean(-1).sqrt().log1p()
    p=scores.softmax(-1)
    entropy=-(p*p.clamp_min(1e-9).log()).sum(-1)
    return torch.stack([first,gap,rms(base),rms(candidate),rms(candidate-base),
        F.cosine_similarity(base,candidate,dim=-1),entropy,torch.full_like(first,fill)],-1)


class UtilityGate(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,1))
        self.register_buffer('mean',torch.zeros(8));self.register_buffer('std',torch.ones(8))

    def forward(self,x):return self.net((x-self.mean)/self.std).squeeze(-1)

    @property
    def tensor_bytes(self):return sum(x.numel()*x.element_size() for x in list(self.parameters())+list(self.buffers()))


def utility_loss(logits,base_error,candidate_error):
    # Weighted binary regret surrogate: harmful acceptance and useful rejection
    # are weighted by the true magnitude of their output MSE difference.
    gain=base_error-candidate_error
    weights=gain.abs()
    return (F.binary_cross_entropy_with_logits(logits,(gain>0).float(),reduction='none')*weights).sum()/weights.sum().clamp_min(1e-8)


class UtilityRouter(RejectionRouter):
    def __init__(self,bank,gate,threshold=.5,hamming=1):
        super().__init__(bank,hamming=hamming)
        self.gate=gate;self.threshold=threshold;self.gate_evaluations=0

    def query(self,q,base,allowed=True):
        self.budget.advance()
        if not allowed or not self.bank.entries or not self.budget.available:return base,False,None
        if self.hamming is not None:
            self.prechecks+=1;sig=signature(q);self.sketch_comparisons+=len(self.sketch)
            if min((sig^int(s)).bit_count() for s in self.sketch)>self.hamming:return base,False,None
        assert self.budget.consume();self.attempts+=1;self.comparisons+=len(self.bank.entries)
        keys=torch.stack([e['key'] for e in self.bank.entries])
        scores=F.normalize(keys,dim=-1)@F.normalize(q.float(),dim=-1)
        idx=int(scores.argmax());entry=self.bank.entries[idx];value=unpack(entry['payload'])
        if self.bank.payload=='residual':value=base+value
        elif self.bank.payload=='anchored':value=value+entry['anchor']
        self.gate_evaluations+=1
        f=features(scores,base,value,len(self.bank.entries)/max(1,self.bank.capacity))
        probability=round(float(self.gate(f).sigmoid()),5)
        if probability<self.threshold:return base,False,None
        self.applied+=1
        return value,True,int(entry['meta'][0])-1
