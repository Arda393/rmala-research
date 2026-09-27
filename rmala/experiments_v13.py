"""Matched retraining of the post gate with a protected-harm penalty."""
import argparse,copy,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed,safe_errors
from .experiments_v5 import projected
from .experiments_v7 import cache,probabilities,mask_for
from .utility_gate import UtilityGate
from .protected_loss import protected_utility_loss
from .cheap_policy import CheapSketchRouter


def train(cs,init,multiplier):
    torch.manual_seed(init+2000);gate=UtilityGate();cs=[c for c in cs if c['bank'].entries]
    x=torch.cat([c['utility_features'] for c in cs]);b=torch.cat([c['base_error'] for c in cs]);v=torch.cat([c['out_error'] for c in cs])
    protected=torch.cat([~c['rare']|~c['stored'] for c in cs])
    gate.mean.copy_(x.mean(0));gate.std.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.Adam(gate.parameters(),lr=.01);rng=torch.Generator().manual_seed(init+3000);trace=[]
    with torch.enable_grad():
        for step in range(300):
            ids=torch.randperm(len(x),generator=rng)[:512]
            loss=protected_utility_loss(gate(x[ids]),b[ids],v[ids],protected[ids],multiplier)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
            if step in [0,299]:trace.append(dict(step=step,loss=float(loss.detach())))
    return gate.requires_grad_(False),dict(multiplier=multiplier,steps=300,examples=len(x),
        protected_harm_examples=int((protected&(v>b)).sum()),useful_examples=int((v<b).sum()),trace=trace)


def replay(c,p,t,h):
    if t>1:return torch.zeros(len(p),dtype=torch.bool),0,c['base_error']
    mask,n=mask_for(c,p,t,h);return mask,n,torch.where(mask,c['out_error'],c['base_error'])


def select(cal,val,gate):
    pool=torch.tensor([p for c in cal if c['bank'].entries for p in probabilities(gate,c)])
    thresholds=sorted(set([0.,.25,.5,.75,.9,.95,.99,1.00001]+[float(pool.quantile(q)) for q in [.5,.9,.95,.99]]))
    vp=[probabilities(gate,c) for c in val];grid=[]
    for h in [0,1,2,4]:
        for t in thresholds:
            rs=[replay(c,p,t,h) for c,p in zip(val,vp)]
            grid.append(dict(threshold=t,hamming=h,safe=all(safe_errors(c,r[2]) for c,r in zip(val,rs)),
                mse=sum(float(r[2].mean()) for r in rs)/len(rs),reads=sum(r[1] for r in rs)))
    best=min([g for g in grid if g['safe']],key=lambda g:(g['mse'],g['reads'],-g['threshold'],g['hamming']))
    return best,grid


def evaluate(c,gate,choice):
    router=CheapSketchRouter(c['bank'],gate,choice,choice['hamming']);ys=[];applied=[]
    for i,(q,b) in enumerate(zip(c['q'],c['base'])):
        y,a,_=router.query(q,b);ys.append(y);applied.append(a)
        assert router.attempts<=int(.05*(i+1)+1e-9)
    expected,reads,err=replay(c,probabilities(gate,c),choice['threshold'],choice['hamming'])
    actual=(torch.stack(ys)-c['y']).square().mean(-1);mask=torch.tensor(applied)
    assert reads==router.attempts and torch.equal(mask,expected)
    assert torch.allclose(actual,err,atol=1e-5,rtol=1e-5)
    nb=float(c['base_error'][c['noisy']].mean());nm=float(actual[c['noisy']].mean())
    harmful=mask&(actual>c['base_error']+1e-9);protected=~c['rare']|~c['stored']
    return dict(mse=float(actual.mean()),baseline_mse=float(c['base_error'].mean()),noisy_mse=nm,noisy_baseline_mse=nb,
        noisy_gain_percent=100*(nb-nm)/max(nb,1e-12),subgroup_preservation_pass=safe_errors(c,actual),
        attempted_reads=reads,applied_corrections=router.applied,harmful_acceptances=int(harmful.sum()),
        protected_harmful_acceptances=int((harmful&protected).sum()),
        protected_harm_sum=float((actual-c['base_error'])[harmful&protected].sum()),
        useful_applications=int((mask&(actual<c['base_error']-1e-9)).sum()),
        sketch_comparisons=router.sketch_comparisons,full_key_comparisons=router.comparisons,
        post_linear_ops=router.gate_evaluations*288,post_static_bytes=gate.tensor_bytes,
        total_tensor_bytes=router.bytes,writes=c['bank'].write_budget.used)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');wp=Path('runs/v8_reference/utility_gates.pt')
    states=torch.load(cp,map_location='cpu');weights=torch.load(wp,map_location='cpu')
    train_seeds=list(range(2010,2026));cal_seeds=list(range(2040,2048));val_seeds=list(range(2060,2080));test_seeds=list(range(2100,2120))
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];logs=[];saved={};count=0
    with torch.no_grad():
        for mode in ['raw_fp32','raw_int8']:
            bases={(n,s):mixed(n,s,mode) for n in names for s in train_seeds+cal_seeds+val_seeds}
            selected=[]
            for init in [41001,41002,41003]:
                model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))])
                cs={(n,s):cache(projected(copy.deepcopy(b),n,s,'learned_context',model)) for (n,s),b in bases.items()}
                training=[cs[n,s] for n in names for s in train_seeds];cal=[cs[n,s] for n in names for s in cal_seeds];val=[cs[n,s] for n in names for s in val_seeds]
                frozen=UtilityGate();frozen.load_state_dict(weights[str((mode,'positive_only',init,'shared'))]);arms={}
                for arm,mult in [('frozen_v8',None),('retrained_standard',1.),('protected_harm4',4.)]:
                    gate,log=(frozen,dict(frozen=True)) if mult is None else train(training,init,mult)
                    choice,grid=select(cal,val,gate);arms[arm]=(gate,choice)
                    logs.append(dict(mode=mode,initialization=init,arm=arm,training=log,selected=choice,grid=grid))
                    saved[str((mode,init,arm))]=gate.state_dict()
                selected.append((init,model,arms));print(json.dumps(dict(mode=mode,init=init,selected={a:c for a,(_,c) in arms.items()})),flush=True)
            tests={(n,s):mixed(n,s,mode) for n in names for s in test_seeds}
            for init,model,arms in selected:
                for (n,s),b in tests.items():
                    c=cache(projected(copy.deepcopy(b),n,s,'learned_context',model))
                    for arm,(gate,choice) in arms.items():
                        row=evaluate(c,gate,choice);row.update(workload=n,seed=s,mode=mode,initialization=init,arm=arm)
                        assert row['total_tensor_bytes']<=2048 and row['writes']<=25
                        with (out/'comparisons.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                        count+=1
                print(json.dumps(dict(mode=mode,init=init,rows=count)),flush=True)
    torch.save(saved,out/'post_gates.pt');(out/'choices.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(dict(rows=count,status='completed',seconds=time.perf_counter()-start,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp]},train_seeds=train_seeds,calibration=cal_seeds,
        validation=val_seeds,test=test_seeds,harm_multiplier=4.,production_ready=False,
        limitations=['Matched initialization/minibatches/300 steps for standard vs harm4; multiplier fixed before evaluation.',
        'Protected=ordinary OR missing labels used only in training and evaluation, never inference.',
        'All three arms recalibrate thresholds/Hamming on same fresh data; frozen_v8 refers to weights only.',
        'No learned pre-gate; one compact sketch precheck and post gate. No full-model FLOP or speed claim.',
        'Frozen projections, synthetic static banks, no LM/100M training.']),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
