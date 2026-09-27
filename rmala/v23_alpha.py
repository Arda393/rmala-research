"""Train contribution size from observables; targets enter training only."""
import torch
from .utility_gate import UtilityGate,features


def scalar_optimum(base,candidate,target):
    d=candidate-base
    den=d.square().sum()
    if float(den)==0.:return .5
    return float((((target-base)*d).sum()/den).clamp(0.,1.))


def tensor_blend(base,candidate,alpha):
    a=alpha.unsqueeze(-1)
    value=base+a*(candidate-base)
    return torch.where(a==0,base,torch.where(a==1,candidate,value))


def train(training,init,steps=1000):
    x=torch.cat([z['c']['utility_features'][z['eligible']] for z in training])
    b=torch.cat([z['c']['base'][z['eligible']] for z in training])
    m=torch.cat([z['c']['output'][z['eligible']] for z in training])
    y=torch.cat([z['c']['y'][z['eligible']] for z in training]);assert len(x)>0
    scalar=scalar_optimum(b,m,y)
    torch.manual_seed(init+23000);head=UtilityGate()
    head.mean.copy_(x.mean(0));head.std.copy_(x.std(0).clamp_min(.01))
    with torch.no_grad():head.net[-1].weight.zero_();head.net[-1].bias.zero_()
    opt=torch.optim.Adam(head.parameters(),lr=.001);rng=torch.Generator().manual_seed(init+24000);trace=[]
    with torch.enable_grad():
        for step in range(1,steps+1):
            ids=torch.randperm(len(x),generator=rng)[:512]
            loss=(tensor_blend(b[ids],m[ids],head(x[ids]).sigmoid())-y[ids]).square().mean()
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
            if step==1 or step%100==0:trace.append(dict(step=step,mse=float(loss.detach())))
    return head.requires_grad_(False),scalar,dict(examples=len(x),steps=steps,lr=.001,initial_alpha=.5,scalar=scalar,trace=trace)


def predict(c,mask,arm,head,scalar,live_values=None):
    a=torch.zeros(len(mask))
    if arm!='query':
        a[mask]={'full':1.,'half':.5,'scalar':scalar}[arm];return a
    keys=None
    if live_values is not None and mask.any():
        keys=torch.nn.functional.normalize(torch.stack([e['key'] for e in c['bank'].entries]),dim=-1)
    for i in mask.nonzero().flatten().tolist():
        f=c['utility_features'][i]
        if keys is not None:
            scores=keys@torch.nn.functional.normalize(c['q'][i].float(),dim=-1)
            f=features(scores,c['base'][i],live_values[i],len(keys)/max(1,c['bank'].capacity))
        a[i]=round(float(head(f).sigmoid()),5)
    return a
