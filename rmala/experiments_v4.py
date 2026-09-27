"""Mixed-query, subgroup-constrained validation. Test labels never route queries."""
import argparse
import json
import time
from pathlib import Path
import torch
from .experiments_v3 import dataset,cached_mask,evaluate


def mixed(name,seed,mode):
    clean=dataset(name,seed,mode,False,True)
    noisy=dataset(name,seed,mode,True,True)
    assert len(clean['bank'].entries)==len(noisy['bank'].entries)
    for a,b in zip(clean['bank'].entries,noisy['bank'].entries):
        assert torch.equal(a['key'],b['key']) and torch.equal(a['payload'][0],b['payload'][0])
    n=len(clean['q']);order=torch.randperm(2*n,generator=torch.Generator().manual_seed(seed+120000))
    out={'bank':clean['bank']}
    for k,v in clean.items():
        if k=='bank':continue
        if isinstance(v,torch.Tensor):out[k]=torch.cat([v,noisy[k]])[order]
        else:
            joined=v+noisy[k];out[k]=[joined[int(i)] for i in order]
    out['noisy']=torch.cat([torch.zeros(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool)])[order]
    return out


def groups(c):
    for name,mask in [('all',torch.ones(len(c['q']),dtype=torch.bool)),('clean',~c['noisy']),('noisy',c['noisy'])]:
        for sub,condition in [('ordinary',~c['rare']),('missing',~c['stored'])]:
            yield name+'_'+sub,mask&condition


def safe_errors(c,errors,tolerance=1e-9):
    # Applied per validation seed AND per regime, not only pooled averages.
    return all(not mask.any() or float((errors[mask]-c['base_error'][mask]).mean())<=tolerance
               for _,mask in groups(c))


def select(cal,val,constrained):
    scores=torch.tensor([s for c in cal for s in c['score'] if s>=-1])
    thresholds=[.9,.95,.98,.99,.995,.999,.9999,1.,1.00001]
    if len(scores):thresholds += [float(scores.quantile(p)) for p in [.5,.9,.95,.99]]
    candidates=[]
    for h in [0,1,2,4]:
        for t in sorted(set(thresholds)):
            total=attempts=applied=0;safe=True
            for c in val:
                mse,reads,mask=cached_mask(c,t,h)
                err=torch.where(mask,c['out_error'],c['base_error'])
                safe=safe and safe_errors(c,err)
                total+=mse;attempts+=reads;applied+=int(mask.sum())
            candidates.append(dict(similarity=t,hamming=h,validation_mse=total/len(val),
                validation_attempts=attempts,validation_applied=applied,validation_safe=safe))
    eligible=[c for c in candidates if not constrained or c['validation_safe']]
    assert eligible
    best=min(eligible,key=lambda c:(c['validation_mse'],c['validation_attempts'],c['similarity'],c['hamming']))
    best=dict(best,active_feasible_candidates=sum(c['validation_safe'] and c['validation_applied']>0 for c in candidates),
        candidate_count=len(candidates),gate=None,gate_threshold=.5)
    return best,candidates


def test_context(c,choice,arm):
    result=evaluate(c,choice,'gla' if arm=='gla' else 'precheck')
    _,_,mask=cached_mask(c,choice['similarity'],choice['hamming'])
    if arm=='gla':mask=torch.zeros_like(mask)
    err=torch.where(mask,c['out_error'],c['base_error'])
    result['subgroup_preservation_pass']=safe_errors(c,err)
    for name,g in [('clean',~c['noisy']),('noisy',c['noisy'])]:
        result[name+'_mse']=float(err[g].mean())
        result[name+'_baseline_mse']=float(c['base_error'][g].mean())
        result[name+'_applied']=int(mask[g].sum())
        result[name+'_false_accepts']=int((mask&g&~c['stored']).sum())
    for name,g in groups(c):
        result[name+'_mse']=float(err[g].mean()) if g.any() else None
        result[name+'_baseline_mse']=float(c['base_error'][g].mean()) if g.any() else None
    return result


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);start=time.perf_counter();rows=[];choices=[];candidates=[]
    with torch.no_grad():
        for name in ['dense_random','rare_details','correlated_dense','repeated_onehot']:
            for mode in ['raw_fp32','raw_int8','residual_fp32']:
                cal=[mixed(name,s,mode) for s in range(410,414)]
                val=[mixed(name,s,mode) for s in [420,421]]
                selected={'gla':dict(similarity=2.,hamming=None,gate=None,gate_threshold=.5)}
                for arm in ['unconstrained','constrained']:
                    choice,grid=select(cal,val,arm=='constrained');selected[arm]=choice
                    choices.append(dict(workload=name,mode=mode,arm=arm,**choice))
                    candidates.append(dict(workload=name,mode=mode,arm=arm,grid=grid))
                # All selection is finished before test contexts are constructed.
                for seed in range(500,510):
                    c=mixed(name,seed,mode)
                    for arm,choice in selected.items():
                        row=dict(workload=name,mode=mode,arm=arm,seed=seed,**test_context(c,choice,arm))
                        rows.append(row)
                        with (out/'comparisons.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(dict(workload=name,mode=mode,rows=len(rows))),flush=True)
    (out/'choices.json').write_text(json.dumps(choices,indent=2))
    (out/'validation_grid.json').write_text(json.dumps(candidates,indent=2))
    summary=dict(status='completed',rows=len(rows),seconds=time.perf_counter()-start,production_ready=False,
        calibration_seeds=list(range(410,414)),validation_seeds=[420,421],test_seeds=list(range(500,510)),
        protocol='1280 mixed clean/noisy queries, one shared prefix budget; one threshold per workload/storage, no regime label to router.',
        preservation='Ordinary and missing MSE <= baseline + 1e-9 numerical tolerance, per validation seed and regime.',
        limitations=['Synthetic frozen bank; same source identities recur across clean/noisy queries.',
        'No learned identity representation or LM integration in this stage.',
        'Budget applies to bank plus sketch, excluding shared base state, Python overhead and transient tensors.',
        'Zero corrections are not a success; noisy-query improvement must be reported separately.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
