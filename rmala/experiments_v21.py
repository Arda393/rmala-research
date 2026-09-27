"""V14 continuation with matched utility and NIL/hard-negative training."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed,safe_errors
from .experiments_v5 import projected
from .experiments_v7 import cache,probabilities
from .experiments_v13 import replay,evaluate
from .utility_gate import UtilityGate
from .nil_gate import match_targets
from .v21_gate import train,fine_cuts

ARMS=['frozen_v14','frozen_fine','utility_fine','nil_hard_fine']


def select(cal,val,base,gate,choice):
    pool=[]
    for c in cal:
        if not c['bank'].entries:continue
        mask,_,_=replay(c,probabilities(base,c),choice['threshold'],choice['hamming'])
        pool.extend(p for p,m in zip(probabilities(gate,c),mask) if m)
    cuts=fine_cuts(pool);grid=[]
    cached=[]
    for c in val:
        p=torch.tensor(probabilities(gate,c))
        eligible,reads,_=replay(c,p.tolist(),0.,choice['hamming'])
        cached.append((p,eligible,reads))
    for t in cuts:
        safe=True;total=0.;reads=0;accept=0
        for c,(p,eligible,n) in zip(val,cached):
            mask=eligible&(p>=t)
            err=torch.where(mask,c['out_error'],c['base_error'])
            safe=safe and safe_errors(c,err);total+=float(err.mean())
            reads+=n if t<=1 else 0;accept+=int(mask.sum())
        grid.append(dict(threshold=t,hamming=choice['hamming'],safe=safe,mse=total/len(val),reads=reads,acceptances=accept))
    best=min([g for g in grid if g['safe']],key=lambda g:(g['mse'],g['reads'],-g['threshold']))
    return best,grid,len(pool)


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');wp=Path('runs/v14_reference/post_gates.pt');ch=wp.with_name('choices.json')
    states=torch.load(cp,map_location='cpu');weights=torch.load(wp,map_location='cpu');old=json.loads(ch.read_text())
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];selected=[];logs=[];saved={}
    with torch.no_grad():
        for init in [41001,41002,41003]:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))]);model.requires_grad_(False)
            def make(n,s):return cache(projected(mixed(n,s,'raw_int8'),n,s,'learned_context',model))
            cs=[make(n,s) for n in names[:2] for s in range(4600,4728)]
            cal=[make(n,s) for n in names for s in range(4740,4756)]
            val=[make(n,s) for n in names for s in range(4800,4900)]
            base=UtilityGate();base.load_state_dict(weights[str(('raw_int8',init,1000))]);base.requires_grad_(False)
            choice=next(c['selected'] for c in old if c['mode']=='raw_int8' and c['initialization']==init and c['step']==1000)
            arms={'frozen_v14':(base,choice)}
            for arm in ARMS[1:]:
                gate,info=(base,dict(frozen=True)) if arm=='frozen_fine' else train(cs,base,init,arm=='nil_hard_fine')
                best,grid,pool=select(cal,val,base,gate,choice)
                arms[arm]=(gate,best);saved[str((init,arm))]=gate.state_dict()
                logs.append(dict(initialization=init,arm=arm,selected=best,grid=grid,calibration_pool=pool,training=info))
                print(json.dumps(dict(initialization=init,arm=arm,selected=best)),flush=True)
            selected.append((init,model,arms));del cs,cal,val
        torch.save(saved,out/'post_gates.pt');(out/'choices.json').write_text(json.dumps(logs,indent=2))
        count=0
        with (out/'comparisons.jsonl').open('w') as f:
            for init,model,arms in selected:
                for n in names:
                    for seed in range(5000,5100):
                        c=cache(projected(mixed(n,seed,'raw_int8'),n,seed,'learned_context',model))
                        for arm,(gate,choice) in arms.items():
                            row=evaluate(c,gate,choice)
                            mask,_,_=replay(c,probabilities(gate,c),choice['threshold'],choice['hamming'])
                            correct=match_targets(c['ids'],c['selected_ids'])
                            row.update(correct_identity_acceptances=int((mask&correct).sum()),wrong_identity_acceptances=int((mask&~correct).sum()),
                                missing_acceptances=int((mask&~c['stored']).sum()),missing_queries=int((~c['stored']).sum()),
                                correct_candidates=int(correct.sum()),mode='raw_int8',workload=n,seed=seed,initialization=init,arm=arm)
                            assert row['total_tensor_bytes']<=2048 and row['writes']<=25
                            f.write(json.dumps(row)+'\n');count+=1
                    f.flush();print(json.dumps(dict(initialization=init,workload=n,rows=count)),flush=True)
    assert count==4800
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,seconds=time.perf_counter()-start,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,ch]},training=[4600,4727],calibration=[4740,4755],validation=[4800,4899],test=[5000,5099],
        production_ready=False,limitations=['Synthetic frozen banks; no real text, 100M or full-model speed claim.',
        'Continues V14 weights and normalization; no extra inference head or identity input.',
        'Missing, hard-wrong and useful-correct strata may overlap; sampling is deliberately reweighted, not population unbiased.',
        'Wrong top candidate means reject candidate, not necessarily no correct entry anywhere.',
        'Hamming setting fixed to V14. Fine threshold candidates use calibration only; validation selects before fresh test.',
        'Closed memory and near-zero useful acceptance do not count as successful correction.']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
