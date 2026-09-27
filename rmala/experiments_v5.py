"""Learned 16D retrieval projections with an explicitly synthetic context control."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .experiments_v2 import workload
from .experiments_v4 import mixed,select,test_context
from .budget_memory import unpack
from .rejection import signature


def attributes(seed,n=512):
    # Observable, non-unique namespace attribute. Shared by many identities.
    rng=torch.Generator().manual_seed(seed+230000)
    codebook=torch.randn(32,16,generator=rng)
    namespaces=torch.randint(32,(n,),generator=rng)
    return codebook,namespaces


def inputs(keys,context):
    semantic=keys-keys.mean(-1,keepdim=True)
    return torch.cat([semantic,context],-1)


def train_projection(contextual):
    torch.manual_seed(31007)
    model=nn.Linear(32,16,bias=False)
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    rng=torch.Generator().manual_seed(31008)
    sources=[];contexts=[]
    for seed in range(610,618):
        k,_,_=workload('dense_random',seed,n=512)
        cb,ns=attributes(seed);sources.append(k);contexts.append(cb[ns])
    k=torch.cat(sources);ctx=torch.cat(contexts)
    losses=[]
    with torch.enable_grad():
        for step in range(240):
            ids=torch.randperm(len(k),generator=rng)[:128]
            key=k[ids];c=ctx[ids] if contextual else torch.zeros_like(ctx[ids])
            q=(key+.03*torch.randn(key.shape,generator=rng)).clamp_min(.001)
            qc=c+.05*torch.randn(c.shape,generator=rng) if contextual else c
            a=F.normalize(model(inputs(q,qc)),dim=-1)
            b=F.normalize(model(inputs(key,c)),dim=-1)
            loss=F.cross_entropy(a@b.T/.07,torch.arange(len(ids)))
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
    model.requires_grad_(False)
    return model,dict(initial_loss=losses[0],final_loss=losses[-1],steps=len(losses),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()))


def projected(c,name,seed,method,model):
    if method in ['legacy_tie','strict_tie']:return c
    source,_,_=workload(name,seed,n=512)
    cb,ns=attributes(seed)
    rng=torch.Generator().manual_seed(seed+240000)
    # Data generation uses identity only to construct repeated observations.
    # No membership flag, target value or unique identity code enters the model.
    qns=torch.randint(32,(len(c['q']),),generator=rng)
    valid=c['ids']>=0;qns[valid]=ns[c['ids'][valid]]
    qc=cb[qns]+.05*torch.randn(len(qns),16,generator=rng)
    sc=cb[ns]
    if method=='learned_semantic':qc=torch.zeros_like(qc);sc=torch.zeros_like(sc)
    q=F.normalize(model(inputs(c['q'],qc)),dim=-1)
    sk=F.normalize(model(inputs(source,sc)),dim=-1)
    for e in c['bank'].entries:e['key']=sk[int(e['meta'][0])-1].clone()
    c['q']=q
    if c['bank'].entries:
        keys=torch.stack([e['key'] for e in c['bank'].entries])
        scores,idx=(q@keys.T).max(-1)
        values=torch.stack([unpack(e['payload']) for e in c['bank'].entries])[idx]
        c['output']=c['base']+values if c['bank'].payload=='residual' else values
        c['score']=[round(float(x),5) for x in scores]
        codes=[signature(k) for k in keys]
        c['distance']=[min((signature(a)^s).bit_count() for s in codes) for a in q]
        c['selected_ids']=torch.tensor([int(e['meta'][0])-1 for e in c['bank'].entries])[idx]
        c['out_error']=(c['output']-c['y']).square().mean(-1)
    return c


def choose(cal,val,strict):
    choice,grid=select(cal,val,True)
    if strict:
        feasible=[x for x in grid if x['validation_safe']]
        best=min(feasible,key=lambda x:(x['validation_mse'],x['validation_attempts'],-x['similarity'],x['hamming']))
        choice.update(best)
    return choice


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    started=time.perf_counter()
    semantic,sa=train_projection(False);contextual,ca=train_projection(True)
    torch.manual_seed(31007);fixed=nn.Linear(32,16,bias=False).requires_grad_(False)
    models={'learned_semantic':semantic,'learned_context':contextual,'fixed_context':fixed}
    torch.save({k:m.state_dict() for k,m in models.items()},out/'projections.pt')
    (out/'training.json').write_text(json.dumps(dict(semantic=sa,context=ca),indent=2))
    choices=[];count=0
    with torch.no_grad():
        for name in ['dense_random','rare_details','correlated_dense','repeated_onehot']:
            for mode in ['raw_fp32','raw_int8','residual_fp32']:
                for method in ['legacy_tie','strict_tie','learned_semantic','fixed_context','learned_context']:
                    def make(seed):return projected(mixed(name,seed,mode),name,seed,method,models.get(method))
                    cal=[make(s) for s in range(610,618)]
                    val=[make(s) for s in range(620,630)]
                    choice=choose(cal,val,method!='legacy_tie')
                    choices.append(dict(workload=name,mode=mode,method=method,**choice))
                    for seed in range(700,720):
                        c=make(seed);row=test_context(c,choice,'constrained')
                        row.update(workload=name,mode=mode,method=method,seed=seed,
                            projection_parameter_bytes=2048 if method in models else 0,
                            projection_query_macs=len(c['q'])*32*16 if method in models else 0,
                            projection_source_macs=512*32*16 if method in models else 0)
                        with (out/'comparisons.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                        count+=1
                    print(json.dumps(dict(workload=name,mode=mode,method=method,rows=count)),flush=True)
    (out/'choices.json').write_text(json.dumps(choices,indent=2))
    summary=dict(status='completed',rows=count,seconds=time.perf_counter()-started,production_ready=False,
        train_calibration_seeds=list(range(610,618)),validation_seeds=list(range(620,630)),test_seeds=list(range(700,720)),
        limitations=['Context is a synthetic observable namespace with 32 categories, not real text or unique identity.',
        'Fixed-context control separates added input information from learning benefit.',
        'Original context-free difficult task is retained; context arms are a different information setting.',
        'Projection weights are 2048 shared static bytes outside the 2048-byte dynamic bank ceiling.',
        'Base GLA and write admission unchanged; learned projections only replace retrieval keys.',
        'Training loss is contrastive matching, not language-model loss. No 100M training.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
