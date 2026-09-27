"""Causal, strictly budgeted reference memory with explicit tensor byte accounting.

This is a research backend, not an efficient GPU kernel. Auxiliary candidate
buffers and reconstruction anchors count against the SAME byte budget as raw KV.
Int8 uses actual int8 payload tensors plus one fp32 scale per vector.
"""
from collections import deque
import math
import torch
from torch.nn import functional as F


class PrefixBudget:
    """At every prefix t, events <= floor(rate*t). Unused credit carries over."""
    def __init__(self,rate):
        if not 0<=rate<=1: raise ValueError("Rate must be in [0,1]")
        self.rate,self.opportunities,self.used=rate,0,0

    def advance(self): self.opportunities+=1

    @property
    def available(self):
        return self.used < math.floor(self.rate*self.opportunities+1e-9)

    def consume(self):
        if not self.available: return False
        self.used+=1
        return True


def pack(value,int8=False):
    if not int8: return value.float().clone(),None
    if value.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("Actual int8 storage is inference-only; no fake quantization gradients")
    scale=value.detach().float().abs().max().clamp_min(1e-8)/127
    return (value.detach()/scale).round().clamp(-127,127).to(torch.int8),scale


def unpack(pair):
    value,scale=pair
    return value.float() if scale is None else value.float()*scale


def packed_bytes(pair):
    return sum(x.numel()*x.element_size() for x in pair if x is not None)


class BudgetMemory:
    def __init__(self,key_dim,value_dim,byte_budget=2048,payload="residual",int8=False,
                 write_rate=.05,read_rate=.05,tau=.3,adaptive=True,history=64,
                 half_life=None,candidates=0,delay=4,topk=8,temperature=.1):
        if payload not in {"raw","residual","anchored"}: raise ValueError("Bad payload")
        if byte_budget<0 or candidates<0 or delay<1 or topk<1 or temperature<=0:
            raise ValueError("Invalid memory configuration")
        self.key_dim,self.value_dim=key_dim,value_dim
        self.byte_budget,self.payload,self.int8=byte_budget,payload,int8
        self.write_budget,self.read_budget=PrefixBudget(write_rate),PrefixBudget(read_rate)
        self.tau,self.adaptive=tau,adaptive
        self.history=deque(maxlen=history)
        self.half_life,self.candidate_capacity,self.delay=half_life,candidates,delay
        self.topk,self.temperature=topk,temperature
        # 3 int64 metadata fields: token id, insertion time, last read time.
        self.entry_bytes=4*key_dim+(value_dim+4 if int8 else 4*value_dim)+24
        if payload=="anchored": self.entry_bytes+=4*value_dim
        self.candidate_bytes=4*(key_dim+value_dim)+24
        reserve=candidates*self.candidate_bytes
        if reserve>byte_budget: raise ValueError("Candidates exceed total byte budget")
        self.capacity=(byte_budget-reserve)//self.entry_bytes
        self.entries=[];self.candidates=[];self.time=0
        self.comparisons=self.queries=self.delayed_writes=self.candidate_checks=0
        self.peak_bytes=0

    @property
    def tensor_bytes(self):
        bank=sum(e['key'].numel()*e['key'].element_size()+packed_bytes(e['payload'])+
                 (0 if e['anchor'] is None else e['anchor'].numel()*e['anchor'].element_size())+
                 e['meta'].numel()*e['meta'].element_size() for e in self.entries)
        pending=sum(c['key'].numel()*c['key'].element_size()+c['value'].numel()*c['value'].element_size()+
                    c['meta'].numel()*c['meta'].element_size() for c in self.candidates)
        return bank+pending

    def _account(self):
        self.peak_bytes=max(self.peak_bytes,self.tensor_bytes)
        if self.tensor_bytes>self.byte_budget: raise AssertionError("Memory byte budget exceeded")

    def threshold(self):
        if not self.adaptive or len(self.history)<8: return self.tau
        # Strictly historical scores, never a quantile over future tokens.
        scores=sorted(self.history)
        i=min(len(scores)-1,int((1-self.write_budget.rate)*len(scores)))
        return max(self.tau,scores[i])

    def _write(self,key,value,estimate,token_id):
        if not self.capacity or not self.write_budget.consume(): return False
        while len(self.entries)>=self.capacity:
            victim=min(range(len(self.entries)),key=lambda i:int(self.entries[i]['meta'][2]))
            self.entries.pop(victim)
        residual=value.float()-estimate.float()
        payload=value.float() if self.payload=="raw" else residual
        self.entries.append(dict(key=key.float().clone(),payload=pack(payload,self.int8),
            anchor=estimate.float().clone() if self.payload=="anchored" else None,
            meta=torch.tensor([token_id,self.time,self.time],dtype=torch.int64,device=key.device)))
        return True

    def observe(self,key,value,estimate,state_read):
        """One source token. state_read(key) uses only the current causal state."""
        self.time+=1;self.write_budget.advance()
        error=float(((value.float()-estimate).norm()/value.float().norm().clamp_min(1e-6)).detach())
        threshold=self.threshold()
        choices=[]
        if error>=threshold: choices.append((error,None,key,value,estimate,self.time))
        # Do not pay revalidation compute while no write credit is available.
        if self.write_budget.available and self.capacity:
            for i,c in enumerate(self.candidates):
                if self.time-int(c['meta'][0])<self.delay: continue
                reconstructed=state_read(c['key'])
                score=float(((c['value']-reconstructed).norm()/c['value'].norm().clamp_min(1e-6)).detach())
                self.candidate_checks+=1
                if score>=threshold:
                    choices.append((score,i,c['key'],c['value'],reconstructed,int(c['meta'][0])))
        wrote_current=False
        if choices and self.write_budget.available and self.capacity:
            _,candidate,key_to_write,val,rec,token_id=max(choices,key=lambda c:c[0])
            if self._write(key_to_write,val,rec,token_id):
                if candidate is not None:
                    self.candidates.pop(candidate);self.delayed_writes+=1
                else: wrote_current=True
        # Bounded FIFO admission; oldest pending values remain until written or
        # until low-error mature candidates can be checked and released.
        if self.write_budget.available:
            retained=[]
            for c in self.candidates:
                if self.time-int(c['meta'][0])>=self.delay:
                    rec=state_read(c['key']);self.candidate_checks+=1
                    score=float(((c['value']-rec).norm()/c['value'].norm().clamp_min(1e-6)).detach())
                    if score<threshold: continue
                retained.append(c)
            self.candidates=retained
        if not wrote_current and len(self.candidates)<self.candidate_capacity:
            self.candidates.append(dict(key=key.float().clone(),value=value.float().clone(),
                meta=torch.tensor([self.time,self.time,self.time],device=key.device,dtype=torch.int64)))
        self.history.append(error);self._account()
        return error

    def gate_features(self,query,denominator,base):
        # No bank similarity scan or unseen target is used by the gate.
        den=torch.as_tensor(denominator,device=query.device).float().clamp_min(1e-6)
        return torch.stack([den.log()/10,query.float().norm().clamp_min(1e-6).log()/5,
            base.float().norm()/math.sqrt(self.value_dim),
            den.new_tensor(len(self.entries)/max(self.capacity,1)),
            den.new_tensor(math.log1p(self.time)/10),
            den.new_tensor((self.time-min((int(e['meta'][1]) for e in self.entries),default=self.time))/max(1,self.time))])

    def retrieve(self,query,base,touch=True):
        if not self.entries: return torch.zeros_like(base)
        keys=torch.stack([e['key'] for e in self.entries])
        scores=F.normalize(keys,dim=-1)@F.normalize(query.float(),dim=-1)
        scores,ids=scores.topk(min(self.topk,len(self.entries)))
        selected=[]
        for i in ids.detach().cpu().tolist():
            entry=self.entries[i];val=unpack(entry['payload'])
            if self.payload=='anchored': val=val+entry['anchor']
            elif self.payload=='residual' and self.half_life is not None:
                val=val*math.exp(-math.log(2)*(self.time-int(entry['meta'][1]))/self.half_life)
            selected.append(val)
            if touch:
                entry['meta']=entry['meta'].clone();entry['meta'][2]=self.time
        value=(scores.div(self.temperature).softmax(0)[:,None]*torch.stack(selected)).sum(0)
        return value if self.payload=='residual' else value-base

    def query(self,query,base,alpha=1.,threshold=.5,hard=True):
        self.read_budget.advance()
        if not self.entries: return base
        if hard:
            if float(torch.as_tensor(alpha).detach())<threshold or not self.read_budget.consume(): return base
        self.queries+=1;self.comparisons+=len(self.entries)
        correction=self.retrieve(query,base)
        return base+correction*(1. if hard else alpha)

    def stats(self):
        return dict(write_rate=self.write_budget.used/max(1,self.write_budget.opportunities),
            retrieval_rate=self.queries/max(1,self.read_budget.opportunities),writes=self.write_budget.used,
            queries=self.queries,comparisons=self.comparisons,peak_tensor_bytes=self.peak_bytes,
            tensor_bytes=self.tensor_bytes,byte_budget=self.byte_budget,capacity=self.capacity,
            delayed_writes=self.delayed_writes,candidate_checks=self.candidate_checks)
