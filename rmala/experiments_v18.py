"""Repeated hard-negative exposure with and without sampling correction."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed
from .experiments_v5 import projected
from .experiments_v7 import cache
from .experiments_v13 import select,evaluate
from .experiments_v14 import train_path
from .experiments_v17 import coverage
from .utility_gate import UtilityGate,utility_loss
from .hard_replay import hard_mask,replay_batch,weighted_utility_loss


def train(cs,init,corrected):
    torch.manual_seed(init+2000);gate=UtilityGate()
    x=torch.cat([c['utility_features'] for c in cs]);b=torch.cat([c['base_error'] for c in cs]);v=torch.cat([c['out_error'] for c in cs])
    hard=torch.cat([hard_mask(c) for c in cs]);gate.mean.copy_(x.mean(0));gate.std.copy_(x.std(0).clamp_min(.01))
    opt=torch.optim.Adam(gate.parameters(),lr=.01);rng=torch.Generator().manual_seed(init+3000);trace=[];exposures=0
    with torch.enable_grad():
        for step in range(1,1001):
            ids,w=replay_batch(hard,rng);exposures+=int(hard[ids].sum())
            loss=weighted_utility_loss(gate(x[ids]),b[ids],v[ids],w if corrected else torch.ones_like(w))
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
            if step==1 or step%100==0:
                with torch.no_grad():full=float(utility_loss(gate(x),b,v))
                trace.append(dict(step=step,original_population_loss=full))
    return gate.requires_grad_(False),dict(hard_examples=int(hard.sum()),hard_draws=exposures,
        total_draws=512000,importance_corrected=corrected,trace=trace)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');ref=Path('runs/v17_reference/post_gates.pt')
    states=torch.load(cp,map_location='cpu');reference=torch.load(ref,map_location='cpu')
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];selected=[];logs=[];saved={};parity=[]
    with torch.no_grad():
        for init in [41001,41002,41003]:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))]);model.requires_grad_(False)
            def make(n,s):return cache(projected(mixed(n,s,'raw_int8'),n,s,'learned_context',model))
            cs=[make(n,s) for n in names[:2] for s in range(2600,2728)]
            cal=[make(n,s) for n in names for s in range(3140,3156)]
            val=[make(n,s) for n in names for s in range(3200,3300)]
            arms={}
            for arm in ['uniform','replay_corrected','replay_unweighted']:
                if arm=='uniform':
                    snapshots,trace=train_path(cs,init,steps=(1000,));gate=snapshots[1000]
                    old=reference[str((init,'train_128'))];diff=max(float((v-old[k]).abs().max()) for k,v in gate.state_dict().items())
                    assert diff==0.;parity.append(dict(initialization=init,max_abs_difference=diff))
                    h=torch.cat([hard_mask(c) for c in cs]);rng=torch.Generator().manual_seed(init+3000)
                    draws=sum(int(h[torch.randperm(len(h),generator=rng)[:512]].sum()) for _ in range(1000))
                    info=dict(trace=trace,hard_examples=int(h.sum()),hard_draws=draws,total_draws=512000)
                else:gate,info=train(cs,init,arm=='replay_corrected')
                choice,grid=select(cal,val,gate);arms[arm]=(gate,choice);saved[str((init,arm))]=gate.state_dict()
                logs.append(dict(initialization=init,arm=arm,selected=choice,grid=grid,training=info,coverage=coverage(cs)))
                print(json.dumps(dict(initialization=init,arm=arm,selected=choice,hard_draws=info['hard_draws'])),flush=True)
            selected.append((init,model,arms));del cs,cal,val
        torch.save(saved,out/'post_gates.pt');(out/'choices.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
        count=0
        with (out/'comparisons.jsonl').open('w',encoding='utf-8') as f:
            for init,model,arms in selected:
                for n in names:
                    for seed in range(3400,3500):
                        c=cache(projected(mixed(n,seed,'raw_int8'),n,seed,'learned_context',model))
                        for arm,(gate,choice) in arms.items():
                            row=evaluate(c,gate,choice);row.update(workload=n,seed=seed,mode='raw_int8',initialization=init,arm=arm)
                            assert row['total_tensor_bytes']<=2048 and row['writes']<=25
                            f.write(json.dumps(row)+'\n');count+=1
                    f.flush();print(json.dumps(dict(initialization=init,workload=n,rows=count)),flush=True)
    assert count==3600
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,seconds=time.perf_counter()-started,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,ref]},reference_parity=parity,
        training=list(range(2600,2728)),calibration=list(range(3140,3156)),validation=list(range(3200,3300)),test=list(range(3400,3500)),
        production_ready=False,limitations=['V17 training reused, fresh selection and test. Uniform weights exactly reproduce V17 train_128.',
        'Replay samples up to 32 hard negatives among 512 per update, without replacement within each batch.',
        'Corrected arm uses self-normalized population/sample-fraction weights; finite-batch ratio is not claimed unbiased.',
        'Unweighted replay deliberately increases hard-negative loss emphasis. No inference labels or extra model parameters.',
        'All arms 1000 updates, same initialization; replay arms share sample stream. No equal wall-time claim.',
        'Synthetic frozen banks and projections; no 100M, real-text or full-model speed result.']),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
