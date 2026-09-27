"""Utility gating against frozen V6 projections; fresh calibration/holdout."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .budget_memory import PrefixBudget
from .experiments_v4 import mixed,safe_errors,test_context,groups
from .experiments_v5 import projected,choose
from .experiments_v6 import INITIALIZATIONS,operation_counts
from .utility_gate import features,UtilityGate,utility_loss,UtilityRouter


def cache(c):
    if not c['bank'].entries:
        c['utility_features']=torch.zeros(len(c['q']),8);return c
    keys=F.normalize(torch.stack([e['key'] for e in c['bank'].entries]),dim=-1)
    # Match the real single-query scoring/features path for calibration audits.
    xs=[]
    for q,b,v in zip(c['q'],c['base'],c['output']):
        scores=keys@F.normalize(q.float(),dim=-1)
        xs.append(features(scores,b,v,len(keys)/max(1,c['bank'].capacity)))
    c['utility_features']=torch.stack(xs)
    return c


def train_gate(cal,init):
    torch.manual_seed(init+2000);gate=UtilityGate()
    cs=[c for c in cal if c['bank'].entries]
    if not cs:return gate.requires_grad_(False),dict(empty_bank=True,steps=0)
    x=torch.cat([c['utility_features'] for c in cs]);b=torch.cat([c['base_error'] for c in cs]);v=torch.cat([c['out_error'] for c in cs])
    gate.mean.copy_(x.mean(0));gate.std.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.Adam(gate.parameters(),lr=.01);rng=torch.Generator().manual_seed(init+3000);log=[]
    with torch.enable_grad():
        for step in range(300):
            ids=torch.randperm(len(x),generator=rng)[:512]
            loss=utility_loss(gate(x[ids]),b[ids],v[ids])
            opt.zero_grad();loss.backward();opt.step()
            if step in [0,299]:log.append(dict(step=step,loss=float(loss.detach())))
    return gate.requires_grad_(False),dict(empty_bank=False,steps=300,log=log)


def mask_for(c,prob,threshold,hamming):
    budget=PrefixBudget(.05);mask=[];attempts=0
    for p,d in zip(prob,c['distance']):
        budget.advance()
        read=bool(c['bank'].entries) and d<=hamming and budget.consume()
        attempts+=int(read);mask.append(read and p>=threshold)
    return torch.tensor(mask),attempts


def probabilities(gate,c):
    # Single-row inference mirrors live router rather than a different BLAS batch.
    return [round(float(gate(x).sigmoid()),5) for x in c['utility_features']]


def select_gate(cal,val,gate):
    probs=[probabilities(gate,c) for c in val]
    pooled=torch.tensor([p for c in cal for p in probabilities(gate,c)])
    thresholds=sorted(set([0.,.25,.5,.75,.9,.95,.99,1.00001]+[float(pooled.quantile(q)) for q in [.5,.9,.95,.99]]))
    grid=[]
    for h in [0,1,2,4]:
        for t in thresholds:
            total=reads=applied=0;safe=True
            for c,p in zip(val,probs):
                mask,n=mask_for(c,p,t,h);err=torch.where(mask,c['out_error'],c['base_error'])
                safe=safe and safe_errors(c,err);total+=float(err.mean());reads+=n;applied+=int(mask.sum())
            grid.append(dict(threshold=t,hamming=h,validation_mse=total/len(val),validation_safe=safe,validation_attempts=reads,validation_applied=applied))
    best=min([x for x in grid if x['validation_safe']],key=lambda x:(x['validation_mse'],x['validation_attempts'],-x['threshold'],x['hamming']))
    return best,grid


def evaluate_gate(c,gate,choice):
    router=UtilityRouter(c['bank'],gate,choice['threshold'],choice['hamming'])
    ys=[];applied=[];correct=[];began=time.perf_counter()
    for i,(q,b) in enumerate(zip(c['q'],c['base'])):
        y,a,idx=router.query(q,b);ys.append(y);applied.append(a);correct.append(a and idx==int(c['ids'][i]) and idx>=0)
        assert router.attempts<=int(.05*(i+1)+1e-9)
    elapsed=time.perf_counter()-began
    err=(torch.stack(ys)-c['y']).square().mean(-1);mask=torch.tensor(applied)
    expected,attempts=mask_for(c,probabilities(gate,c),choice['threshold'],choice['hamming'])
    assert attempts==router.attempts and torch.equal(mask,expected),'Cached/live routing mismatch'
    assert torch.allclose(err,torch.where(mask,c['out_error'],c['base_error']),atol=1e-5,rtol=1e-5)
    missing=~c['stored'];ratio=lambda a,b:float(a)/float(b) if float(b) else None
    row=dict(mse=float(err.mean()),baseline_mse=float(c['base_error'].mean()),queries=len(err),
        attempted_reads=router.attempts,applied_corrections=router.applied,comparisons=router.comparisons,
        prechecks=router.prechecks,sketch_comparisons=router.sketch_comparisons,
        total_tensor_bytes=router.bytes,writes=c['bank'].write_budget.used,capacity=c['bank'].capacity,
        gate_evaluations=router.gate_evaluations,gate_parameter_and_buffer_bytes=gate.tensor_bytes,
        gate_linear_arithmetic_ops=router.gate_evaluations*288,
        seconds=elapsed,subgroup_preservation_pass=safe_errors(c,err),correct_identity_applications=sum(correct),
        false_accept_rate=ratio((mask&missing).sum(),missing.sum()),
        false_reject_rate=ratio((~mask&c['stored']).sum(),c['stored'].sum()),
        useful_read_fraction=ratio((mask&(err<c['base_error'])).sum(),router.attempts),
        harmful_acceptances=int((mask&(err>c['base_error']+1e-9)).sum()),
        useful_rejections=int((~mask&(c['out_error']<c['base_error']-1e-9)).sum()))
    for name,m in [('stored',c['stored']),('missing',missing),('ordinary',~c['rare']),('rare',c['rare']),('near_wrong',c['ids']<0),('clean',~c['noisy']),('noisy',c['noisy'])]+list(groups(c)):
        row[name+'_count']=int(m.sum());row[name+'_mse']=float(err[m].mean()) if m.any() else None
        row[name+'_baseline_mse']=float(c['base_error'][m].mean()) if m.any() else None
    for n,m in [('clean',~c['noisy']),('noisy',c['noisy'])]:
        row[n+'_applied']=int((mask&m).sum());row[n+'_false_accepts']=int((mask&m&missing).sum())
    row.update(operation_counts(row,True))
    assert router.bytes<=2048 and row['writes']<=25
    return row


def run(out,checkpoint):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    states=torch.load(checkpoint,map_location='cpu');models={}
    for init in INITIALIZATIONS:
        for method in ['fixed_context','positive_only']:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str((method,init))]);model.requires_grad_(False)
            models[method,init]=model
    choices=[];training=[];weights={};count=0
    with torch.no_grad():
        for name in ['dense_random','rare_details','correlated_dense','repeated_onehot']:
            for mode in ['raw_fp32','raw_int8','residual_fp32']:
                bases={s:mixed(name,s,mode) for s in list(range(1010,1018))+list(range(1020,1030))}
                selected=[]
                for (method,init),model in models.items():
                    def make(s):return cache(projected(copy.deepcopy(bases[s]),name,s,'learned_context',model))
                    cal=[make(s) for s in range(1010,1018)];val=[make(s) for s in range(1020,1030)]
                    gate,log=train_gate(cal,init);gchoice,grid=select_gate(cal,val,gate);schoice=choose(cal,val,True)
                    info=dict(workload=name,mode=mode,projection=method,initialization=init)
                    choices.append(dict(**info,utility=gchoice,similarity=schoice,grid=grid));training.append(dict(**info,**log))
                    weights[str((name,mode,method,init))]=gate.state_dict()
                    selected.append((method,init,model,gate,gchoice,schoice))
                tests={s:mixed(name,s,mode) for s in range(1100,1120)}
                for method,init,model,gate,gchoice,schoice in selected:
                    for seed,b in tests.items():
                        c=cache(projected(copy.deepcopy(b),name,seed,'learned_context',model))
                        for arm in ['similarity','utility']:
                            row=test_context(c,schoice,'constrained') if arm=='similarity' else evaluate_gate(c,gate,gchoice)
                            row.update(operation_counts(row,True))
                            if arm=='similarity':row.update(gate_evaluations=0,gate_parameter_and_buffer_bytes=0,gate_linear_arithmetic_ops=0)
                            row.update(workload=name,mode=mode,projection=method,initialization=init,seed=seed,arm=arm)
                            with (out/'comparisons.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                            count+=1
                    print(json.dumps(dict(workload=name,mode=mode,projection=method,init=init,rows=count)),flush=True)
    torch.save(weights,out/'utility_gates.pt')
    (out/'choices.json').write_text(json.dumps(choices,indent=2));(out/'training.json').write_text(json.dumps(training,indent=2))
    summary=dict(status='completed',rows=count,seconds=time.perf_counter()-start,production_ready=False,
        projection_checkpoint=str(checkpoint),gate_initializations=INITIALIZATIONS,
        train_calibration_seeds=list(range(1010,1018)),validation_seeds=list(range(1020,1030)),test_seeds=list(range(1100,1120)),
        limitations=['Frozen V6 projections, synthetic context, not LM training.',
        'Post-retrieval rejection still consumes a read; utility training caches use additional offline searches.',
        'Gate linear operation counts exclude tanh, sigmoid, feature extraction and normalization.',
        'Projection arithmetic estimates exclude gate costs and full model; gate costs reported separately.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--checkpoint',default='runs/v6_reference/projections.pt');a=p.parse_args();run(a.out,a.checkpoint)
