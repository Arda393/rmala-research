"""Continue matched standard training; select checkpoint without test access."""
import argparse,copy,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from .utility_gate import UtilityGate,utility_loss
from .experiments_v4 import mixed
from .experiments_v5 import projected
from .experiments_v7 import cache,probabilities
from .experiments_v13 import select,evaluate
from .ranking_audit import average_precision


def train_path(cs,init,steps=(300,1000,3000)):
    if not steps or min(steps)<1:raise ValueError('Positive checkpoint steps required')
    torch.manual_seed(init+2000);gate=UtilityGate();cs=[c for c in cs if c['bank'].entries]
    x=torch.cat([c['utility_features'] for c in cs]);b=torch.cat([c['base_error'] for c in cs]);v=torch.cat([c['out_error'] for c in cs])
    gate.mean.copy_(x.mean(0));gate.std.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.Adam(gate.parameters(),lr=.01);rng=torch.Generator().manual_seed(init+3000)
    snapshots={};trace=[]
    with torch.enable_grad():
        for step in range(1,max(steps)+1):
            ids=torch.randperm(len(x),generator=rng)[:512]
            loss=utility_loss(gate(x[ids]),b[ids],v[ids]);assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step()
            if step==1 or step%100==0 or step in steps:
                with torch.no_grad():full=float(utility_loss(gate(x),b,v))
                trace.append(dict(step=step,training_full_loss=full))
            if step in steps:
                frozen=UtilityGate();frozen.load_state_dict(copy.deepcopy(gate.state_dict()));snapshots[step]=frozen.requires_grad_(False)
    return snapshots,trace


def separation(c,gate):
    p=probabilities(gate,c);positive=(c['out_error']<c['base_error']-1e-9).tolist()
    harmful=((c['out_error']>c['base_error']+1e-9)&(~c['rare']|~c['stored'])).tolist()
    mean=lambda mask:sum(x for x,m in zip(p,mask) if m)/sum(mask) if any(mask) else None
    return dict(positive_rate=sum(positive)/len(p),average_precision=average_precision(p,positive),
        useful_probability_mean=mean(positive),protected_harmful_probability_mean=mean(harmful),
        calibration_loss=float(utility_loss(gate(c['utility_features']),c['base_error'],c['out_error'])))


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');ref=Path('runs/v13_reference/post_gates.pt')
    states=torch.load(cp,map_location='cpu');reference=torch.load(ref,map_location='cpu')
    train_seeds=list(range(2010,2026));cal_seeds=list(range(2240,2248));val_seeds=list(range(2260,2280));test_seeds=list(range(2300,2320))
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];logs=[];diagnostics=[];saved={};count=0;parity=[]
    with torch.no_grad():
        for mode in ['raw_fp32','raw_int8']:
            bases={(n,s):mixed(n,s,mode) for n in names for s in train_seeds+cal_seeds+val_seeds};selected=[]
            for init in [41001,41002,41003]:
                model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))])
                cs={(n,s):cache(projected(copy.deepcopy(b),n,s,'learned_context',model)) for (n,s),b in bases.items()}
                training=[cs[n,s] for n in names for s in train_seeds];cal=[cs[n,s] for n in names for s in cal_seeds];val=[cs[n,s] for n in names for s in val_seeds]
                gates,trace=train_path(training,init);old=reference[str((mode,init,'retrained_standard'))]
                diffs=[]
                for k,v in gates[300].state_dict().items():
                    assert torch.allclose(v,old[k],atol=1e-7,rtol=1e-6),k
                    diffs.append(float((v-old[k]).abs().max()))
                parity.append(dict(mode=mode,initialization=init,max_abs_difference=max(diffs)))
                choices={}
                for step,gate in gates.items():
                    choice,grid=select(cal,val,gate);choices[step]=choice
                    logs.append(dict(mode=mode,initialization=init,step=step,selected=choice,grid=grid,trace=trace))
                    saved[str((mode,init,step))]=gate.state_dict()
                    for n in names[:2]:
                        for s in cal_seeds:diagnostics.append(dict(mode=mode,initialization=init,step=step,workload=n,seed=s,**separation(cs[n,s],gate)))
                chosen=min(choices,key=lambda step:(choices[step]['mse'],choices[step]['reads'],step))
                selected.append((init,model,gates,choices,chosen));print(json.dumps(dict(mode=mode,init=init,chosen_step=chosen)),flush=True)
            tests={(n,s):mixed(n,s,mode) for n in names for s in test_seeds}
            for init,model,gates,choices,chosen in selected:
                for (n,s),b in tests.items():
                    c=cache(projected(copy.deepcopy(b),n,s,'learned_context',model))
                    for step,gate in gates.items():
                        row=evaluate(c,gate,choices[step]);row.update(workload=n,mode=mode,initialization=init,seed=s,step=step,chosen_step=chosen)
                        assert row['total_tensor_bytes']<=2048 and row['writes']<=25
                        with (out/'comparisons.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                        count+=1
                print(json.dumps(dict(mode=mode,init=init,rows=count)),flush=True)
    torch.save(saved,out/'post_gates.pt')
    for n,data in [('choices.json',logs),('calibration_diagnostics.json',diagnostics)]:
        (out/n).write_text(json.dumps(data,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,seconds=time.perf_counter()-started,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,ref]},reference_parity=parity,
        train_seeds=train_seeds,calibration=cal_seeds,validation=val_seeds,test=test_seeds,production_ready=False,
        limitations=['Same standard-loss training trajectory at 300/1000/3000 steps, same data/order/constant Adam lr0.01.',
        'V13 training contexts reused deliberately; separate fresh calibration/validation/test.',
        'Checkpoint chosen using validation MSE/read count only; all test curves are secondary, not a new choice.',
        'Frozen projections and synthetic banks; no new architecture, no full-model speed or LM claim.']),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
