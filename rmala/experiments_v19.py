"""Original-key sketch verification with frozen V14 gates and fresh holdout."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed,safe_errors
from .experiments_v5 import projected
from .experiments_v7 import cache,probabilities
from .experiments_v13 import replay,evaluate as reference_evaluate
from .raw_verifier import RawSketchVerifier
from .rejection import signature
from .utility_gate import UtilityGate


def make(name,seed,model):
    c=mixed(name,seed,'raw_int8')
    raw=c['q'].clone();sketch=torch.tensor([signature(e['key']) for e in c['bank'].entries],dtype=torch.int16)
    c=cache(projected(c,name,seed,'learned_context',model));c['raw_q']=raw;c['raw_sketch']=sketch
    # Source admission/capacity is unchanged: both sketches fit for all entries.
    assert c['bank'].tensor_bytes+4*len(c['bank'].entries)<=2048
    if c['bank'].entries:
        ids=[int(e['meta'][0])-1 for e in c['bank'].entries]
        c['raw_distance']=torch.tensor([(signature(q)^int(sketch[ids.index(int(i))])).bit_count() for q,i in zip(raw,c['selected_ids'])])
    else:c['raw_distance']=torch.full((len(raw),),16)
    return c


def cached(c,gate,choice,h):
    if h==-1:return torch.zeros(len(c['q']),dtype=torch.bool),0,c['base_error']
    mask,reads,_=replay(c,probabilities(gate,c),choice['threshold'],choice['hamming'])
    if h is not None:mask=mask&(c['raw_distance']<=h)
    return mask,reads,torch.where(mask,c['out_error'],c['base_error'])


def evaluate(c,gate,choice,h):
    router=RawSketchVerifier(c['bank'],gate,choice,c['raw_sketch'] if h is not None else None,h)
    ys=[];applied=[]
    for i,(q,b,raw) in enumerate(zip(c['q'],c['base'],c['raw_q'])):
        y,a,_=router.query(q,b,raw);ys.append(y);applied.append(a)
        assert router.attempts<=int(.05*(i+1)+1e-9)
    mask=torch.tensor(applied);actual=(torch.stack(ys)-c['y']).square().mean(-1)
    expected,reads,err=cached(c,gate,choice,h)
    assert reads==router.attempts and torch.equal(mask,expected)
    assert torch.allclose(actual,err,atol=1e-5,rtol=1e-5)
    harmful=mask&(actual>c['base_error']+1e-9);protected=~c['rare']|~c['stored']
    nb=float(c['base_error'][c['noisy']].mean());nm=float(actual[c['noisy']].mean())
    row=dict(mse=float(actual.mean()),baseline_mse=float(c['base_error'].mean()),noisy_mse=nm,noisy_baseline_mse=nb,
        noisy_gain_percent=100*(nb-nm)/max(nb,1e-12),subgroup_preservation_pass=safe_errors(c,actual),
        attempted_reads=reads,applied_corrections=router.applied,protected_harmful_acceptances=int((harmful&protected).sum()),
        protected_harm_sum=float((actual-c['base_error'])[harmful&protected].sum()),full_key_comparisons=router.comparisons,
        post_linear_ops=router.gate_evaluations*288,total_tensor_bytes=router.bytes,writes=c['bank'].write_budget.used,
        verifier_checks=router.verifier_checks,verifier_rejections=router.verifier_rejections,
        verifier_bytes=router.raw_sketch.numel()*2,metadata_checks=router.metadata_checks,raw_signature_elements=router.raw_signature_elements)
    if h is None:
        old=reference_evaluate(c,gate,choice)
        for k in old:
            if k in row:assert old[k]==row[k],k
    return row


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');wp=Path('runs/v14_reference/post_gates.pt');ch=wp.with_name('choices.json')
    states=torch.load(cp,map_location='cpu');weights=torch.load(wp,map_location='cpu');choices=json.loads(ch.read_text())
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];selected=[];logs=[]
    with torch.no_grad():
        for init in [41001,41002,41003]:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))]);model.requires_grad_(False)
            gate=UtilityGate()
            gate.load_state_dict(weights[str(('raw_int8',init,1000))]);gate.requires_grad_(False)
            choice=next(c['selected'] for c in choices if c['mode']=='raw_int8' and c['initialization']==init and c['step']==1000)
            val=[make(n,s,model) for n in names for s in range(3600,3700)]
            # Cached post-gate masks are shared across verifier candidates.
            base=[cached(c,gate,choice,None) for c in val];grid=[]
            for h in [-1,0,1,2,3,4,15]:
                errs=[c['base_error'] if h==-1 else torch.where(m&(c['raw_distance']<=h),c['out_error'],c['base_error']) for c,(m,_,_) in zip(val,base)]
                grid.append(dict(raw_hamming=h,safe=all(safe_errors(c,e) for c,e in zip(val,errs)),mse=sum(float(e.mean()) for e in errs)/len(errs),reads=0 if h==-1 else sum(r[1] for r in base)))
            best=min([g for g in grid if g['safe']],key=lambda g:(g['mse'],g['reads'],g['raw_hamming']))
            selected.append((init,model,gate,choice,best['raw_hamming']));logs.append(dict(initialization=init,frozen_post_choice=choice,selected=best,grid=grid))
            print(json.dumps(logs[-1]),flush=True);del val,base,errs
        (out/'choices.json').write_text(json.dumps(logs,indent=2),encoding='utf-8');count=0
        with (out/'comparisons.jsonl').open('w',encoding='utf-8') as f:
            for init,model,gate,choice,h in selected:
                for n in names:
                    for seed in range(3800,3900):
                        c=make(n,seed,model)
                        for arm,cut in [('frozen_v14',None),('raw_verifier',h)]:
                            row=evaluate(c,gate,choice,cut);row.update(workload=n,seed=seed,initialization=init,arm=arm,mode='raw_int8',raw_hamming=cut)
                            f.write(json.dumps(row)+'\n');count+=1
                    f.flush();print(json.dumps(dict(initialization=init,workload=n,rows=count)),flush=True)
    assert count==2400
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,seconds=time.perf_counter()-start,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,ch]},validation=list(range(3600,3700)),test=list(range(3800,3900)),
        production_ready=False,limitations=['No retraining; frozen V14 int8 1000-step gates/thresholds, all three initializations.',
        '15-bit original-key sketch is not collision-free membership; only observable source/query keys used.',
        'Two extra bytes per stored entry included in 2KB. Capacity/admission unchanged and asserted to fit.',
        'Verification after post-gate acceptance; rejected reads still consume budget and gate compute.',
        'Raw query vectors are offline test inputs, not persistent bank state. Signature and metadata scan operations reported.',
        'Fresh validation chooses verifier radius, then unseen test; no real-text or full-model speed claim.']),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
