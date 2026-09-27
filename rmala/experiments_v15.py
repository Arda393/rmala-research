"""Fresh confirmation with frozen V14 int8 gates and thresholds; no selection."""
import argparse, copy, hashlib, json, time
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed
from .experiments_v5 import projected
from .experiments_v7 import cache
from .experiments_v13 import evaluate
from .utility_gate import UtilityGate


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt')
    wp=Path('runs/v14_reference/post_gates.pt');ch=wp.with_name('choices.json')
    states=torch.load(cp,map_location='cpu');weights=torch.load(wp,map_location='cpu')
    choices=json.loads(ch.read_text(encoding='utf-8'));mode='raw_int8';policies=[]
    # Freeze every policy before constructing any new test context.
    for init in [41001,41002,41003]:
        available=[c for c in choices if c['mode']==mode and c['initialization']==init]
        chosen=min(available,key=lambda c:(c['selected']['mse'],c['selected']['reads'],c['step']))
        assert chosen['step']==1000
        for step in [300,1000]:
            matches=[c for c in available if c['step']==step];assert len(matches)==1
            policies.append(dict(mode=mode,initialization=init,step=step,choice=matches[0]['selected']))
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,ch]}
    (out/'frozen_policies.json').write_text(json.dumps(dict(provenance=provenance,policies=policies),indent=2),encoding='utf-8')
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];count=0
    with torch.no_grad(),(out/'comparisons.jsonl').open('w',encoding='utf-8') as f:
        for init in [41001,41002,41003]:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))]);arms=[]
            for p in policies:
                if p['initialization']!=init:continue
                gate=UtilityGate();gate.load_state_dict(weights[str((mode,init,p['step']))]);gate.requires_grad_(False)
                arms.append((p,gate))
            for name in names:
                for seed in range(2400,2500):
                    c=cache(projected(mixed(name,seed,mode),name,seed,'learned_context',model))
                    for p,gate in arms:
                        row=evaluate(c,gate,p['choice'])
                        row.update(workload=name,seed=seed,mode=mode,initialization=init,step=p['step'])
                        assert row['writes']<=25 and row['total_tensor_bytes']<=2048
                        f.write(json.dumps(row)+'\n');count+=1
                f.flush();print(json.dumps(dict(initialization=init,workload=name,rows=count)),flush=True)
    assert count==2400
    assert provenance=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,ch]}
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,seconds=time.perf_counter()-start,
        provenance=provenance,test=list(range(2400,2500)),retraining=False,recalibration=False,
        production_ready=False,limitations=['Frozen V14 int8 1000 vs 300 steps on new synthetic contexts.',
        'Same 100 contexts reused across three projection/gate initializations; not 300 independent training runs.',
        'No full-model FLOP, throughput, real-text or residual-over-raw claim.']),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
