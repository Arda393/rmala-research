"""V14 fine tuning: utility plus explicit wrong-candidate rejection."""
import copy
import torch
from torch.nn import functional as F
from .utility_gate import utility_loss
from .nil_gate import match_targets


def label_pools(c):
    correct=match_targets(c['ids'],c['selected_ids'])
    missing=(~c['stored'])&(~correct)
    hard=(~correct)&((torch.tensor(c['distance'])==0)|(c['utility_features'][:,0]>=.99))
    positive=correct&(c['out_error']<c['base_error'])
    return missing,hard,positive


def fine_cuts(scores):
    cuts={0.,.25,.5,.75,.9,.95,.99,1.,1.00001}
    cuts.update(round(.99+i*.0001,5) for i in range(101))
    if scores:
        x=torch.tensor(scores)
        cuts.update(round(float(x.quantile(i/100)),5) for i in range(101))
    return sorted(cuts)


def train(cs,base,init,auxiliary,steps=1000):
    g=copy.deepcopy(base).requires_grad_(True)
    x=torch.cat([c['utility_features'] for c in cs]);b=torch.cat([c['base_error'] for c in cs]);v=torch.cat([c['out_error'] for c in cs])
    masks=[torch.cat([label_pools(c)[i] for c in cs]) for i in range(3)]
    pools=[m.nonzero().flatten() for m in masks]
    assert all(len(p)>0 for p in pools), 'Missing required training stratum'
    rng=torch.Generator().manual_seed(init+21000);opt=torch.optim.Adam(g.parameters(),lr=.001)
    trace=[];draws=[0,0,0]
    with torch.enable_grad():
        for step in range(1,steps+1):
            ids=torch.randperm(len(x),generator=rng)[:512]
            sampled=[p[torch.randint(len(p),(32,),generator=rng)] for p in pools]
            for i in range(3):draws[i]+=len(sampled[i])
            loss=utility_loss(g(x[ids]),b[ids],v[ids])
            extra=sum(F.binary_cross_entropy_with_logits(g(x[s]),torch.full((len(s),),float(i==2))) for i,s in enumerate(sampled))/3
            if auxiliary:loss=loss+.1*extra
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step()
            if step==1 or step%100==0:trace.append(dict(step=step,loss=float(loss.detach()),auxiliary_loss=float(extra.detach())))
    return g.requires_grad_(False),dict(steps=steps,lr=.001,auxiliary_weight=.1 if auxiliary else 0.,
        examples=len(x),pool_counts=[len(p) for p in pools],stratum_draws=draws,trace=trace)
