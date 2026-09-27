"""Isolate the empty-bank calibration pool bug with frozen V10 weights."""
import argparse,copy,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed,safe_errors
from .experiments_v5 import projected
from .experiments_v10 import prepare,probs,replay
from .utility_gate import UtilityGate
from .fast_policy import FrozenFastRouter


def candidates(probability_groups,populated,active_only):
    values=[p for ps,active in zip(probability_groups,populated) if active or not active_only for p in ps]
    if not values:return [0.,1.00001]
    pool=torch.tensor(values)
    return sorted(set([0.,.25,.5,.75,.9,1.00001]+[float(pool.quantile(q)) for q in [.5,.75,.9,.95,.99]]))


def select(cal,val,gate,post,active_only):
    thresholds=candidates([probs(gate,c) for c in cal],[bool(c['bank'].entries) for c in cal],active_only)
    vp=[probs(gate,c) for c in val];grid=[]
    for t in thresholds:
        rs=[replay(c,p,t,post) for c,p in zip(val,vp)]
        grid.append(dict(threshold=t,safe=all(safe_errors(c,r[2]) for c,r in zip(val,rs)),
                         mse=sum(float(r[2].mean()) for r in rs)/len(rs),reads=sum(r[1] for r in rs)))
    return min([g for g in grid if g['safe']],key=lambda g:(g['mse'],g['reads'],-g['threshold'])),grid


def evaluate(c,post,choice,pre,t):
    router=FrozenFastRouter(c['bank'],post,choice,pre,t);ys=[];applied=[]
    for i,(q,b) in enumerate(zip(c['q'],c['base'])):
        y,a,_=router.query(q,b);ys.append(y);applied.append(a)
        assert router.attempts<=int(.05*(i+1)+1e-9)
    expected,reads,err=replay(c,probs(pre,c),t,choice)
    actual=(torch.stack(ys)-c['y']).square().mean(-1)
    assert torch.equal(expected,torch.tensor(applied)) and torch.allclose(err,actual,atol=1e-5,rtol=1e-5)
    assert router.attempts==reads or choice['threshold']>1
    nb=float(c['base_error'][c['noisy']].mean());nm=float(actual[c['noisy']].mean())
    return dict(mse=float(actual.mean()),baseline_mse=float(c['base_error'].mean()),noisy_mse=nm,noisy_baseline_mse=nb,
        noisy_gain_percent=100*(nb-nm)/max(nb,1e-12),subgroup_preservation_pass=safe_errors(c,actual),
        attempted_reads=router.attempts,applied_corrections=router.applied,pre_evaluations=router.pre_evaluations,
        pre_linear_ops=router.pre_evaluations*288,total_tensor_bytes=router.bytes,writes=c['bank'].write_budget.used)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    paths=[Path('runs/v6_reference/projections.pt'),Path('runs/v8_reference/utility_gates.pt'),Path('runs/v10_reference/pre_gates.pt'),Path('runs/v10_reference/choices.json')]
    states,posts,pres=[torch.load(p,map_location='cpu') for p in paths[:3]];old=json.loads(paths[3].read_text())
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];logs=[];count=0
    with torch.no_grad():
        for mode in ['raw_fp32','raw_int8']:
            bases={(n,s):mixed(n,s,mode) for n in names for s in list(range(1610,1618))+list(range(1620,1630))}
            selected=[]
            for init in [41001,41002,41003]:
                model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))])
                post=UtilityGate();post.load_state_dict(posts[str((mode,'positive_only',init,'shared'))])
                pre=UtilityGate();pre.load_state_dict(pres[str((mode,init))])
                choice=next(x['post'] for x in old if x['mode']==mode and x['initialization']==init)
                cs={(n,s):prepare(projected(copy.deepcopy(b),n,s,'learned_context',model),post) for (n,s),b in bases.items()}
                cal=[cs[n,s] for n in names for s in range(1610,1618)];val=[cs[n,s] for n in names for s in range(1620,1630)]
                ts={}
                for arm in ['mixed_pool','active_pool']:
                    best,grid=select(cal,val,pre,choice,arm=='active_pool');ts[arm]=best['threshold']
                    logs.append(dict(mode=mode,initialization=init,arm=arm,selected=best,grid=grid))
                ts['post_only']=0.;selected.append((init,model,post,pre,choice,ts))
                print(json.dumps(dict(mode=mode,init=init,selected=ts)),flush=True)
            tests={(n,s):mixed(n,s,mode) for n in names for s in range(1700,1720)}
            for init,model,post,pre,choice,ts in selected:
                for (n,s),b in tests.items():
                    c=prepare(projected(copy.deepcopy(b),n,s,'learned_context',model),post)
                    for arm,t in ts.items():
                        row=evaluate(c,post,choice,pre,t);row.update(workload=n,seed=s,mode=mode,initialization=init,arm=arm)
                        assert row['total_tensor_bytes']<=2048 and row['writes']<=25
                        with (out/'comparisons.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                        count+=1
                print(json.dumps(dict(mode=mode,init=init,rows=count)),flush=True)
    (out/'choices.json').write_text(json.dumps(logs,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(rows=count,status='completed',seconds=time.perf_counter()-start,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},calibration=list(range(1610,1618)),
        validation=list(range(1620,1630)),test=list(range(1700,1720)),new_training=False,
        limitations=['Only threshold candidate pool differs between mixed_pool and active_pool; frozen weights and same grid quantiles.',
        'post_only is unselected control; frozen post threshold>1 fast path skips provably useless reads.',
        'Fresh holdout tests an implementation correction, not a new trained model or full model FLOPs.']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
