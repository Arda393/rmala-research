"""Cheap sketch selection vs learned preselection on fresh held-out contexts."""
import argparse,copy,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed,safe_errors
from .experiments_v5 import projected
from .experiments_v10 import prepare,probs,replay
from .threshold_audit import select
from .utility_gate import UtilityGate
from .fast_policy import FrozenFastRouter
from .cheap_policy import CheapSketchRouter


def select_cheap(val,post):
    grid=[]
    for h in [-1]+list(range(post['hamming']+1)):
        choice=dict(post,hamming=h);rs=[replay(c,[1.]*len(c['q']),0.,choice) for c in val]
        grid.append(dict(hamming=h,safe=all(safe_errors(c,r[2]) for c,r in zip(val,rs)),
                         mse=sum(float(r[2].mean()) for r in rs)/len(rs),reads=sum(r[1] for r in rs)))
    return min([g for g in grid if g['safe']],key=lambda g:(g['mse'],g['reads'],g['hamming'])),grid


def evaluate(c,post,choice,pre,arm,setting):
    if arm=='cheap_sketch':
        router=CheapSketchRouter(c['bank'],post,choice,setting)
        expected,reads,err=replay(c,[1.]*len(c['q']),0.,dict(choice,hamming=min(choice['hamming'],setting)))
        pre_bytes=0
    else:
        router=FrozenFastRouter(c['bank'],post,choice,pre,setting)
        expected,reads,err=replay(c,probs(pre,c),setting,choice)
        pre_bytes=pre.tensor_bytes if arm=='learned_pre' and 0<setting<=1 and choice['threshold']<=1 else 0
    ys=[];applied=[]
    for i,(q,b) in enumerate(zip(c['q'],c['base'])):
        y,a,_=router.query(q,b);ys.append(y);applied.append(a)
        assert router.attempts<=int(.05*(i+1)+1e-9)
    actual=(torch.stack(ys)-c['y']).square().mean(-1)
    assert torch.equal(expected,torch.tensor(applied)) and torch.allclose(err,actual,atol=1e-5,rtol=1e-5)
    assert router.attempts==reads or choice['threshold']>1
    nb=float(c['base_error'][c['noisy']].mean());nm=float(actual[c['noisy']].mean())
    return dict(mse=float(actual.mean()),baseline_mse=float(c['base_error'].mean()),noisy_mse=nm,noisy_baseline_mse=nb,
        noisy_gain_percent=100*(nb-nm)/max(nb,1e-12),subgroup_preservation_pass=safe_errors(c,actual),
        attempted_reads=router.attempts,applied_corrections=router.applied,
        harmful_acceptances=int((torch.tensor(applied)&(actual>c['base_error']+1e-9)).sum()),
        pre_evaluations=router.pre_evaluations,pre_linear_ops=router.pre_evaluations*288,pre_required_static_bytes=pre_bytes,
        sketch_comparisons=router.sketch_comparisons+router.extra_sketch_comparisons,
        full_key_comparisons=router.comparisons,post_evaluations=router.gate_evaluations,
        post_linear_ops=router.gate_evaluations*288,total_tensor_bytes=router.bytes,writes=c['bank'].write_budget.used)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    paths=[Path('runs/v6_reference/projections.pt'),Path('runs/v8_reference/utility_gates.pt'),Path('runs/v10_reference/pre_gates.pt'),Path('runs/v10_reference/choices.json')]
    states,posts,pres=[torch.load(p,map_location='cpu') for p in paths[:3]];old=json.loads(paths[3].read_text())
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];logs=[];count=0
    with torch.no_grad():
        for mode in ['raw_fp32','raw_int8']:
            bases={(n,s):mixed(n,s,mode) for n in names for s in list(range(1810,1818))+list(range(1820,1830))}
            selected=[]
            for init in [41001,41002,41003]:
                model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))])
                post=UtilityGate();post.load_state_dict(posts[str((mode,'positive_only',init,'shared'))])
                pre=UtilityGate();pre.load_state_dict(pres[str((mode,init))])
                choice=next(x['post'] for x in old if x['mode']==mode and x['initialization']==init)
                cs={(n,s):prepare(projected(copy.deepcopy(b),n,s,'learned_context',model),post) for (n,s),b in bases.items()}
                cal=[cs[n,s] for n in names for s in range(1810,1818)];val=[cs[n,s] for n in names for s in range(1820,1830)]
                learned,lg=select(cal,val,pre,choice,True);cheap,cg=select_cheap(val,choice)
                ts=dict(learned_pre=learned['threshold'],cheap_sketch=cheap['hamming'],post_only=0.)
                logs.append(dict(mode=mode,initialization=init,selected=ts,learned_grid=lg,cheap_grid=cg,post=choice))
                selected.append((init,model,post,pre,choice,ts));print(json.dumps(dict(mode=mode,init=init,selected=ts)),flush=True)
            tests={(n,s):mixed(n,s,mode) for n in names for s in range(1900,1920)}
            for init,model,post,pre,choice,ts in selected:
                for (n,s),b in tests.items():
                    c=prepare(projected(copy.deepcopy(b),n,s,'learned_context',model),post)
                    for arm,setting in ts.items():
                        row=evaluate(c,post,choice,pre,arm,setting);row.update(workload=n,seed=s,mode=mode,initialization=init,arm=arm)
                        assert row['total_tensor_bytes']<=2048 and row['writes']<=25
                        with (out/'comparisons.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                        count+=1
                print(json.dumps(dict(mode=mode,init=init,rows=count)),flush=True)
    (out/'choices.json').write_text(json.dumps(logs,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(rows=count,status='completed',seconds=time.perf_counter()-start,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},calibration=list(range(1810,1818)),
        validation=list(range(1820,1830)),test=list(range(1900,1920)),new_training=False,
        limitations=['Frozen learned weights; cheap Hamming cutoff and learned threshold selected separately with same validation safety.',
        'Cheap cutoff folds into existing Hamming scan; no pre neural network or second scan.',
        'Static byte metric means deployment-required pre weights; harness still loads all controls in memory.',
        'Arithmetic/sketch counts exclude feature, normalization and full-model costs; no measured speedup claim.']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
