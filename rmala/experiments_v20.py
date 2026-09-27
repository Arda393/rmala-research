"""Observable-feature audit plus a matched identity/NIL vs utility guard pilot."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .experiments_v4 import mixed,safe_errors
from .experiments_v5 import projected
from .experiments_v7 import cache,probabilities
from .experiments_v13 import replay,evaluate as evaluate_base
from .experiments_v14 import train_path
from .utility_gate import UtilityGate
from .nil_gate import match_targets,GuardedGate
from .cheap_policy import CheapSketchRouter
from .ranking_audit import average_precision


def train_nil(cs,init):
    torch.manual_seed(init+2000);g=UtilityGate()
    x=torch.cat([c['utility_features'] for c in cs]);y=torch.cat([match_targets(c['ids'],c['selected_ids']) for c in cs]).float()
    g.mean.copy_(x.mean(0));g.std.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.Adam(g.parameters(),lr=.01);rng=torch.Generator().manual_seed(init+3000);trace=[]
    with torch.enable_grad():
        for step in range(1,1001):
            ids=torch.randperm(len(x),generator=rng)[:512];loss=F.binary_cross_entropy_with_logits(g(x[ids]),y[ids])
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
            if step==1 or step%100==0:trace.append(dict(step=step,loss=float(loss.detach())))
    return g.requires_grad_(False),dict(examples=len(y),positive_examples=int(y.sum()),trace=trace)


def audit(c,guards):
    labels=match_targets(c['ids'],c['selected_ids']);x=c['utility_features'];buckets={}
    for row,y in zip(x,labels):
        key=tuple(round(float(v),5) for v in row);buckets.setdefault(key,set()).add(bool(y))
    hard=(~labels)&(x[:,0]>=.99)&(torch.tensor(c['distance'])==0)
    result=dict(examples=len(labels),correct_candidates=int(labels.sum()),hard_wrong_candidates=int(hard.sum()),
        conflicting_rounded_feature_buckets=sum(len(v)>1 for v in buckets.values()),scores={})
    for name,g in guards.items():
        ps=probabilities(g,c);result['scores'][name]=dict(match_average_precision=average_precision(ps,labels.tolist()))
    if hard.any() and labels.any():
        # Distance only within this calibration context, standardized with training statistics.
        g=guards['nil_guard'];z=(x-g.mean)/g.std
        ds=torch.cdist(z[hard],z[labels]).min(-1).values
        result['nearest_opposite_label_feature_distance']=dict(min=float(ds.min()),median=float(ds.median()),max=float(ds.max()))
    return result


def select_guard(cal,val,base,guard,choice):
    pool=torch.tensor([p for c in cal if c['bank'].entries for p in probabilities(guard,c)])
    cuts=sorted(set([0.,.25,.5,.75,.9,.95,.99,1.00001]+[float(pool.quantile(q)) for q in [.5,.9,.95,.99]]))
    cached=[replay(c,probabilities(base,c),choice['threshold'],choice['hamming']) for c in val]
    ps=[torch.tensor(probabilities(guard,c)) for c in val];grid=[]
    for t in cuts:
        errs=[c['base_error'] if t>1 else torch.where(mask&(p>=t),c['out_error'],c['base_error']) for c,(mask,_,_),p in zip(val,cached,ps)]
        grid.append(dict(threshold=t,safe=all(safe_errors(c,e) for c,e in zip(val,errs)),mse=sum(float(e.mean()) for e in errs)/len(errs),reads=0 if t>1 else sum(r[1] for r in cached)))
    return min([g for g in grid if g['safe']],key=lambda g:(g['mse'],g['reads'],-g['threshold'])),grid


def evaluate(c,base,guard,choice,threshold):
    combined=GuardedGate(base,guard,choice['threshold'],threshold)
    outer=dict(threshold=1.00001 if threshold>1 else .5,hamming=choice['hamming'])
    router=CheapSketchRouter(c['bank'],combined,outer,choice['hamming']);ys=[];mask=[]
    for i,(q,b) in enumerate(zip(c['q'],c['base'])):
        y,a,_=router.query(q,b);ys.append(y);mask.append(a)
        assert router.attempts<=int(.05*(i+1)+1e-9)
    mask=torch.tensor(mask);actual=(torch.stack(ys)-c['y']).square().mean(-1)
    old,reads,_=replay(c,probabilities(base,c),choice['threshold'],choice['hamming'])
    expected=old&(torch.tensor(probabilities(guard,c))>=threshold)
    if threshold>1:reads=0;expected=torch.zeros_like(old)
    err=torch.where(expected,c['out_error'],c['base_error'])
    assert torch.equal(mask,expected) and router.attempts==reads
    assert torch.allclose(actual,err,atol=1e-5,rtol=1e-5)
    correct=match_targets(c['ids'],c['selected_ids']);harmful=mask&(actual>c['base_error']+1e-9);protected=~c['rare']|~c['stored']
    nb=float(c['base_error'][c['noisy']].mean());nm=float(actual[c['noisy']].mean())
    return dict(mse=float(actual.mean()),noisy_mse=nm,noisy_gain_percent=100*(nb-nm)/max(nb,1e-12),
        subgroup_preservation_pass=safe_errors(c,actual),attempted_reads=reads,applied_corrections=router.applied,
        correct_identity_acceptances=int((mask&correct).sum()),wrong_identity_acceptances=int((mask&~correct).sum()),
        missing_acceptances=int((mask&~c['stored']).sum()),missing_queries=int((~c['stored']).sum()),correct_candidates=int(correct.sum()),
        protected_harmful_acceptances=int((harmful&protected).sum()),protected_harm_sum=float((actual-c['base_error'])[harmful&protected].sum()),
        total_tensor_bytes=router.bytes,writes=c['bank'].write_budget.used,full_key_comparisons=router.comparisons,
        post_linear_ops=(combined.base_evaluations+combined.guard_evaluations)*288,guard_evaluations=combined.guard_evaluations,post_static_bytes=combined.tensor_bytes)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');wp=Path('runs/v14_reference/post_gates.pt');ch=wp.with_name('choices.json')
    states=torch.load(cp,map_location='cpu');weights=torch.load(wp,map_location='cpu');old_choices=json.loads(ch.read_text())
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];selected=[];logs=[];diagnostics=[];saved={}
    with torch.no_grad():
        for init in [41001,41002,41003]:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))]);model.requires_grad_(False)
            def make(n,s):return cache(projected(mixed(n,s,'raw_int8'),n,s,'learned_context',model))
            training=[make(n,s) for n in names[:2] for s in range(4000,4128)]
            cal=[make(n,s) for n in names for s in range(4140,4156)];val=[make(n,s) for n in names for s in range(4200,4300)]
            snapshots,trace=train_path(training,init,steps=(1000,));nil,info=train_nil(training,init)
            guards={'utility_guard':snapshots[1000],'nil_guard':nil}
            base=UtilityGate();base.load_state_dict(weights[str(('raw_int8',init,1000))]);base.requires_grad_(False)
            choice=next(c['selected'] for c in old_choices if c['mode']=='raw_int8' and c['initialization']==init and c['step']==1000)
            # Diagnostic results are recorded without changing protocol, features or labels.
            for n in names[:2]:
                for s,c in zip(range(4140,4156),cal[names.index(n)*16:(names.index(n)+1)*16]):
                    diagnostics.append(dict(workload=n,seed=s,initialization=init,**audit(c,guards)))
            arms={}
            for name,guard in guards.items():
                best,grid=select_guard(cal,val,base,guard,choice);arms[name]=(guard,best['threshold']);saved[str((init,name))]=guard.state_dict()
                logs.append(dict(initialization=init,arm=name,selected=best,grid=grid,base_choice=choice,training=info if name=='nil_guard' else dict(trace=trace)))
                print(json.dumps(dict(initialization=init,arm=name,selected=best)),flush=True)
            selected.append((init,model,base,choice,arms));del training,cal,val
        torch.save(saved,out/'guards.pt')
        (out/'choices.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
        (out/'feature_audit.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8');count=0
        with (out/'comparisons.jsonl').open('w',encoding='utf-8') as f:
            for init,model,base,choice,arms in selected:
                for n in names:
                    for seed in range(4400,4500):
                        c=cache(projected(mixed(n,seed,'raw_int8'),n,seed,'learned_context',model))
                        for arm in ['frozen_v14','utility_guard','nil_guard']:
                            if arm=='frozen_v14':
                                row=evaluate_base(c,base,choice);mask,_,_=replay(c,probabilities(base,c),choice['threshold'],choice['hamming']);correct=match_targets(c['ids'],c['selected_ids'])
                                row.update(correct_identity_acceptances=int((mask&correct).sum()),wrong_identity_acceptances=int((mask&~correct).sum()),missing_acceptances=int((mask&~c['stored']).sum()),missing_queries=int((~c['stored']).sum()),correct_candidates=int(correct.sum()),guard_evaluations=0)
                            else:row=evaluate(c,base,arms[arm][0],choice,arms[arm][1])
                            assert row['total_tensor_bytes']<=2048 and row['writes']<=25
                            row.update(workload=n,seed=seed,initialization=init,arm=arm,mode='raw_int8');f.write(json.dumps(row)+'\n');count+=1
                    f.flush();print(json.dumps(dict(initialization=init,workload=n,rows=count)),flush=True)
    assert count==3600
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,seconds=time.perf_counter()-start,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,ch]},training=list(range(4000,4128)),calibration=list(range(4140,4156)),validation=list(range(4200,4300)),test=list(range(4400,4500)),production_ready=False,
        limitations=['First NIL-inspired pilot, not a CLINK reproduction: no semantic types, candidate masking or new feature inputs.',
        'Same 8 observables, architecture, initialization, batches and 1000 steps for auxiliary utility vs match BCE guards.',
        'Identity labels are used only in training/diagnostics. Gate sees no query identity or bank-membership flag.',
        'Guard examines only the retrieved top candidate; rejection does not prove that no other bank entry is correct.',
        'Both guards retain the frozen V14 utility decision, then filter. Rejected reads still cost budget.',
        'Calibration feature collisions/distances and AP are empirical diagnostics, not an identifiability impossibility proof.',
        'No new context/type information. No real-text or 100M result. Static extra head bytes are separate from 2KB dynamic bank.']),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
