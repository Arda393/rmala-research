"""Paired multi-initialization hard-negative projection experiment."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .experiments_v2 import workload
from .experiments_v4 import mixed,test_context
from .experiments_v5 import attributes,inputs,projected,choose


INITIALIZATIONS=[41001,41002,41003]


def negative_views(key,context,rng):
    nk=(key+.015*torch.randn(key.shape,generator=rng)).clamp_min(.001)
    # Half collide in observable category: deliberately harder than test's 1/32.
    collision=torch.rand(len(key),generator=rng)<.5
    nc=torch.randn(context.shape,generator=rng)
    nc[collision]=context[collision]
    nc=nc+.05*torch.randn(nc.shape,generator=rng)
    return nk,nc,collision


def train(init,hard):
    torch.manual_seed(init);model=nn.Linear(32,16,bias=False)
    rng=torch.Generator().manual_seed(init+100)
    sources=[];contexts=[]
    for seed in range(810,818):
        k,_,_=workload('dense_random',seed,n=512);cb,ns=attributes(seed)
        sources.append(k);contexts.append(cb[ns])
    k=torch.cat(sources);ctx=torch.cat(contexts)
    opt=torch.optim.Adam(model.parameters(),lr=.01);logs=[]
    with torch.enable_grad():
        for step in range(240):
            ids=torch.randperm(len(k),generator=rng)[:128];key=k[ids];c=ctx[ids]
            q=(key+.03*torch.randn(key.shape,generator=rng)).clamp_min(.001)
            qc=c+.05*torch.randn(c.shape,generator=rng)
            # Both arms consume the same RNG sequence for paired training views.
            nk,nc,collision=negative_views(key,c,rng)
            a=F.normalize(model(inputs(q,qc)),dim=-1)
            b=F.normalize(model(inputs(key,c)),dim=-1)
            match=F.cross_entropy(a@b.T/.07,torch.arange(len(ids)))
            wrong=F.normalize(model(inputs(nk,nc)),dim=-1)
            penalty=F.softplus(((wrong*b).sum(-1)-.97)/.02).mean()
            loss=match+(2*penalty if hard else 0)
            opt.zero_grad();loss.backward();opt.step()
            if step in [0,239]:logs.append(dict(step=step,loss=float(loss.detach()),match_loss=float(match.detach()),
                negative_penalty=float(penalty.detach()),collision_count=int(collision.sum())))
    model.requires_grad_(False)
    return model,logs


def operation_counts(row,projected_model):
    # Arithmetic model, not profiler total: multiply/add/div/sqrt each count as one.
    n=row['stored_count']//2;q=row['queries'];reads=row['attempted_reads'];d=16
    projection=q*(32*16+31*16) if projected_model else 0
    preprocessing=q*((16-1)+1+16+3*16) if projected_model else 0
    # Actual router normalizes all bank keys and query again on every search.
    per_read=n*((2*d-1)+3*d)+3*d if n else 0
    score=reads*per_read
    precheck=row['prechecks']*d # mean reduction + division; bit ops separate.
    full=q*per_read
    selective=projection+preprocessing+score+precheck
    dense_same=projection+preprocessing+full
    return dict(query_projection_arithmetic_ops=projection,query_preprocessing_arithmetic_ops=preprocessing,
        router_score_arithmetic_ops=score,precheck_arithmetic_ops=precheck,
        modeled_query_arithmetic_ops=selective,modeled_dense_same_projection_ops=dense_same,
        modeled_query_saving_percent=100*(1-selective/dense_same) if dense_same else 0,
        precheck_integer_code_comparisons=row['sketch_comparisons'],
        projection_parameter_bytes=2048 if projected_model else 0,
        source_projection_arithmetic_ops=512*(32*16+31*16+80) if projected_model else 0)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    started=time.perf_counter();models={};training=[]
    for init in INITIALIZATIONS:
        torch.manual_seed(init);models['fixed_context',init]=nn.Linear(32,16,bias=False).requires_grad_(False)
        for method in ['positive_only','hard_negative']:
            model,logs=train(init,method=='hard_negative');models[method,init]=model
            training.append(dict(method=method,initialization=init,logs=logs))
    torch.save({str(k):m.state_dict() for k,m in models.items()},out/'projections.pt')
    (out/'training.json').write_text(json.dumps(training,indent=2))
    count=0;choices=[]
    with torch.no_grad():
        for name in ['dense_random','rare_details','correlated_dense','repeated_onehot']:
            for mode in ['raw_fp32','raw_int8','residual_fp32']:
                # Build source/query data once. Copies isolate each frozen projection.
                base_cal={s:mixed(name,s,mode) for s in range(810,818)}
                base_val={s:mixed(name,s,mode) for s in range(820,830)}
                base_test=None
                selected=[]
                for (method,init),model in models.items():
                    def transform(c,s):return projected(copy.deepcopy(c),name,s,'learned_context',model)
                    cal=[transform(c,s) for s,c in base_cal.items()]
                    val=[transform(c,s) for s,c in base_val.items()]
                    choice=choose(cal,val,True);selected.append((method,init,model,choice))
                    choices.append(dict(workload=name,mode=mode,method=method,initialization=init,**choice))
                choice=choose(list(base_cal.values()),list(base_val.values()),True)
                selected.append(('context_free',-1,None,choice))
                choices.append(dict(workload=name,mode=mode,method='context_free',initialization=-1,**choice))
                # Only now construct held-out test contexts.
                base_test={s:mixed(name,s,mode) for s in range(900,920)}
                for method,init,model,choice in selected:
                    for seed,base in base_test.items():
                        c=copy.deepcopy(base)
                        if model is not None:c=projected(c,name,seed,'learned_context',model)
                        row=test_context(c,choice,'constrained')
                        row.update(operation_counts(row,model is not None))
                        row.update(workload=name,mode=mode,method=method,initialization=init,seed=seed)
                        with (out/'comparisons.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                        count+=1
                    print(json.dumps(dict(workload=name,mode=mode,method=method,initialization=init,rows=count)),flush=True)
    (out/'choices.json').write_text(json.dumps(choices,indent=2))
    summary=dict(status='completed',rows=count,seconds=time.perf_counter()-started,production_ready=False,
        initializations=INITIALIZATIONS,train_calibration_seeds=list(range(810,818)),
        validation_seeds=list(range(820,830)),test_seeds=list(range(900,920)),
        limitations=['Synthetic observable category context; not LM or real-text context extraction.',
        'Hard negatives collide in category 50% during training versus approximately 1/32 in test.',
        'Arithmetic estimate excludes base GLA/model, payload operations, bit operations, sorting, allocations and memory traffic.',
        'Static projection weights are additional to the dynamic 2048-byte bank ceiling.',
        'No post-test threshold adjustment; no 100M training.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
