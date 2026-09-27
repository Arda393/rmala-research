"""Training-only stratified replay; no diagnostic labels enter inference."""
import torch
from torch.nn import functional as F


def hard_mask(c):
    return ((c['out_error']>c['base_error']+1e-9)&(~c['rare']|~c['stored'])&
        (c['selected_ids']!=c['ids'])&(c['utility_features'][:,0]>=.99)&(torch.tensor(c['distance'])==0))


def replay_batch(hard,rng,batch_size=512,hard_count=32):
    if hard.ndim!=1 or hard.dtype!=torch.bool or len(hard)<batch_size or not 0<hard_count<batch_size:
        raise ValueError('Require boolean mask, sufficient population and valid stratum count')
    h=hard.nonzero().flatten();other=(~hard).nonzero().flatten()
    if not len(h) or not len(other):
        ids=torch.randperm(len(hard),generator=rng)[:batch_size]
        return ids,torch.ones(len(ids))
    nh=min(hard_count,len(h));nr=min(batch_size-nh,len(other));nh=batch_size-nr
    ids=torch.cat([h[torch.randperm(len(h),generator=rng)[:nh]],other[torch.randperm(len(other),generator=rng)[:nr]]])
    # Population fraction / sampled fraction: self-normalized importance loss.
    weights=torch.cat([torch.full((nh,),len(h)*batch_size/(len(hard)*nh)),
                       torch.full((nr,),len(other)*batch_size/(len(hard)*nr))])
    assert len(ids)==batch_size
    return ids,weights


def weighted_utility_loss(logits,base_error,candidate_error,sample_weights):
    gain=base_error-candidate_error;w=gain.abs()*sample_weights
    return (F.binary_cross_entropy_with_logits(logits,(gain>0).float(),reduction='none')*w).sum()/w.sum().clamp_min(1e-8)
