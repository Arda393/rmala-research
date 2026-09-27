"""Disjoint-seed rejection experiment; no learned LM quality claim."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .budget_memory import PrefixBudget,unpack
from .experiments_v2 import context,workload,train_gates
from .rejection import RejectionRouter,signature


def dataset(name,seed,mode,noisy,pre=False):
    # Reserve sketch bytes before source admission; no hidden index allocation.
    entry=108 if mode=='raw_int8' else 152
    reserve=2*(2048//(entry+2)) if pre else 0
    ctx=context(name,seed,mode,.05,2048-reserve,n=512)
    bank=ctx['bank'];rng=torch.Generator().manual_seed(seed+91000)
    q=ctx['k'].clone()
    if noisy: q=(q+.03*torch.randn(q.shape,generator=rng)).clamp_min(.001)
    # Near-wrong keys represent NEW identities, never inserted in source state.
    # Their values are independent, or share the workload's ordinary common value.
    nq=(ctx['k'][:128]+.015*torch.randn(128,16,generator=rng)).clamp_min(.001)
    nv=torch.randn(128,16,generator=rng)
    if name in ['rare_details','correlated_dense']:
        _,source_values,rare=workload(name,seed,n=512)
        common=source_values[~rare].mean(0)
        nv=common+.01*nv
    if name=='repeated_onehot':
        # Deliberately ambiguous new identities, with independent values.
        pass
    q=torch.cat([q,nq]);y=torch.cat([ctx['v'],nv])
    ids=torch.cat([ctx['token_ids'],torch.full((128,),-1)])
    rare=torch.cat([ctx['rare'],torch.zeros(128,dtype=torch.bool)])
    order=torch.randperm(len(q),generator=rng)
    q,y,ids,rare=q[order],y[order],ids[order],rare[order]
    base=q@ctx['s']/(q@ctx['z']).clamp_min(1e-6)[:,None]
    # Evaluation labels: repeated identical source keys share a value identity.
    stored_ids={int(e['meta'][0])-1 for e in bank.entries}
    stored=torch.tensor([int(i) in stored_ids for i in ids])
    features=torch.stack([bank.gate_features(a,a@ctx['z'],b) for a,b in zip(q,base)])
    if bank.entries:
        keys=torch.stack([e['key'] for e in bank.entries])
        sims=F.normalize(q,dim=-1)@F.normalize(keys,dim=-1).T
        score,index=sims.max(-1)
        selected_ids=torch.tensor([int(e['meta'][0])-1 for e in bank.entries])[index]
        vals=torch.stack([unpack(e['payload']) for e in bank.entries])[index]
        output=base+vals if mode=='residual_fp32' else vals
        if name=='repeated_onehot':
            stored=torch.tensor([int(i)>=0 and any(torch.equal(ctx['k'][(ctx['token_ids']==i).nonzero()[0,0]],e['key']) for e in bank.entries) for i in ids])
        sketches=[signature(e['key']) for e in bank.entries]
        distances=[min((signature(a)^s).bit_count() for s in sketches) for a in q]
    else:
        score=torch.full((len(q),),-2.);output=base.clone();selected_ids=torch.full_like(ids,-2)
        distances=[16]*len(q)
    return dict(bank=bank,q=q,y=y,ids=ids,rare=rare,stored=stored,base=base,features=features,
                score=[round(float(s),5) for s in score],distance=distances,output=output,selected_ids=selected_ids,
                base_error=(base-y).square().mean(-1),out_error=(output-y).square().mean(-1))


def cached_mask(c,threshold,hamming=None,gate=None,gate_threshold=.5):
    budget=PrefixBudget(.05);mask=[];attempts=0
    allowed=[True]*len(c['q']) if gate is None else (gate(c['features']).sigmoid().flatten()>=gate_threshold).tolist()
    for score,dist,allow in zip(c['score'],c['distance'],allowed):
        budget.advance()
        attempt=allow and bool(c['bank'].entries) and (hamming is None or dist<=hamming) and budget.consume()
        attempts+=int(attempt)
        mask.append(attempt and score>=threshold)
    mask=torch.tensor(mask)
    return torch.where(mask,c['out_error'],c['base_error']).mean().item(),attempts,mask


def select(cal,val,arm):
    if arm=='gla': return dict(similarity=2.,hamming=None,gate=None,gate_threshold=2.)
    if arm=='existing_gate':
        caches=[(c['features'],c['base'],c['output']-c['base'],c['y']) for c in cal]
        candidates=[]
        for penalty,gate in train_gates(caches,[0.,.2,1.]):
            pred=torch.cat([gate(c['features']).sigmoid().flatten() for c in val])
            for t in sorted(set([.5,1.]+[float(pred.quantile(x)) for x in [0,.5,.9,.99]])):
                result=[cached_mask(c,-2,gate=gate,gate_threshold=t) for c in val]
                candidates.append((sum(r[0] for r in result),sum(r[1] for r in result),penalty,t,gate))
        best=min(candidates,key=lambda x:x[:4])
        return dict(similarity=-2.,hamming=None,gate=best[4],gate_threshold=best[3])
    # Calibration supplies candidate cutoffs; ONLY validation selects a cutoff.
    scores=torch.tensor([s for c in cal for s in c['score'] if s>=-1])
    thresholds=[.9,.95,.98,.99,.995,.999,.9999,1.00001]
    if len(scores): thresholds += [float(scores.quantile(p)) for p in [.5,.9,.95,.99]]
    candidates=[]
    for h in ([0,1,2,4] if arm=='precheck' else [None]):
        for t in sorted(set(thresholds)):
            result=[cached_mask(c,t,h) for c in val]
            candidates.append((sum(r[0] for r in result),sum(r[1] for r in result),t,h))
    best=min(candidates,key=lambda x:x[:3])
    return dict(similarity=best[2],hamming=best[3],gate=None,gate_threshold=.5)


def evaluate(c,choice,arm):
    router=RejectionRouter(c['bank'],similarity=choice['similarity'],hamming=choice['hamming'])
    preds=[];applied=[];correct=[]
    gate=choice['gate'];start=time.perf_counter()
    for i,(q,b) in enumerate(zip(c['q'],c['base'])):
        allowed=arm!='gla' and (gate is None or float(gate(c['features'][i]).sigmoid())>=choice['gate_threshold'])
        y,a,selected=router.query(q,b,allowed)
        preds.append(y);applied.append(a)
        correct.append(a and int(c['ids'][i])>=0 and selected==int(c['ids'][i]))
        assert router.attempts<=int(.05*(i+1)+1e-9)
    elapsed=time.perf_counter()-start
    err=(torch.stack(preds)-c['y']).square().mean(-1)
    mask=torch.tensor(applied);stored=c['stored'];missing=~stored
    cached,attempts,expected=cached_mask(c,choice['similarity'],choice['hamming'],gate,choice['gate_threshold'])
    if arm!='gla':
        assert torch.equal(mask,expected) and attempts==router.attempts
        assert abs(float(err.mean())-cached)<1e-5
    def mean(x,m): return float(x[m].mean()) if m.any() else None
    def rate(a,b): return int(a)/int(b) if int(b) else None
    result=dict(mse=float(err.mean()),baseline_mse=float(c['base_error'].mean()),
        attempted_reads=router.attempts,applied_corrections=router.applied,
        false_accept_rate=rate((mask&missing).sum(),missing.sum()),
        false_reject_rate=rate((~mask&stored).sum(),stored.sum()),
        useful_read_fraction=rate((mask&(err<c['base_error'])).sum(),router.attempts),
        correct_identity_applications=sum(correct),queries=len(err),
        comparisons=router.comparisons,prechecks=router.prechecks,sketch_comparisons=router.sketch_comparisons,
        index_build_key_elements=router.index_build_key_elements,
        precheck_key_elements=router.prechecks*16,total_tensor_bytes=router.bytes,
        writes=c['bank'].write_budget.used,capacity=c['bank'].capacity,seconds=elapsed)
    for name,m in [('stored',stored),('missing',missing),('near_wrong',c['ids']<0),('rare',c['rare']),('ordinary',~c['rare'])]:
        result[name+'_count']=int(m.sum());result[name+'_mse']=mean(err,m)
        result[name+'_baseline_mse']=mean(c['base_error'],m)
    assert result['writes']<=25 and router.bytes<=2048
    return result


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.perf_counter();choices=[];rows=[]
    with torch.no_grad():
        for name in ['dense_random','rare_details','correlated_dense','repeated_onehot']:
            for mode in ['raw_fp32','raw_int8','residual_fp32']:
                for noisy in [False,True]:
                    for pre in [False,True]:
                        cal=[dataset(name,s,mode,noisy,pre) for s in range(210,214)]
                        val=[dataset(name,s,mode,noisy,pre) for s in range(220,222)]
                        arms=['precheck'] if pre else ['gla','existing_gate','postcheck']
                        tests=[dataset(name,s,mode,noisy,pre) for s in range(300,310)]
                        for arm in arms:
                            choice=select(cal,val,arm)
                            info=dict(workload=name,mode=mode,noisy=noisy,arm=arm)
                            saved={k:v for k,v in choice.items() if k!='gate'}
                            if choice['gate'] is not None: saved['gate_state']={k:v.tolist() for k,v in choice['gate'].state_dict().items()}
                            choices.append(dict(**info,**saved))
                            for seed,c in zip(range(300,310),tests):
                                row=dict(**info,seed=seed,**evaluate(c,choice,arm));rows.append(row)
                                with (out/'comparisons.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(json.dumps(dict(workload=name,mode=mode,noisy=noisy,rows=len(rows))),flush=True)
    (out/'choices.json').write_text(json.dumps(choices,indent=2))
    summary=dict(status='completed',production_ready=False,rows=len(rows),elapsed_seconds=time.perf_counter()-started,
        calibration_seeds=list(range(210,214)),validation_seeds=[220,221],test_seeds=list(range(300,310)),
        caveats=['Frozen-bank synthetic inference; not integrated into learned LM attention.',
        '640 queries: 512 original identities plus 128 new near-wrong identities; prevalence is synthetic.',
        'Noisy positives use sigma .03; near-wrong keys use sigma .015, deliberately overlapping.',
        'Existing six-feature gate retrained on fresh calibration with top1 retrieval to isolate routing.',
        'Sketch is a linear scan of 15-bit codes, not O(1) lookup. Python/allocator overhead excluded.',
        'False rejection includes budget-denied reads. Postcheck rejection still consumes a read.',
        'Timing excludes feature construction for old gate; no end-to-end speedup claim.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
