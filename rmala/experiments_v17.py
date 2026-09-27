"""Fixed-update training coverage comparison, with a frozen V14 control."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed
from .experiments_v5 import projected
from .experiments_v7 import cache
from .experiments_v13 import select,evaluate
from .experiments_v14 import train_path
from .utility_gate import UtilityGate


def coverage(cs):
    total=useful=harmful=hard=0
    for c in cs:
        bad=(c['out_error']>c['base_error']+1e-9)&(~c['rare']|~c['stored'])
        wrong=c['selected_ids']!=c['ids']
        difficult=bad&wrong&(c['utility_features'][:,0]>=.99)&(torch.tensor(c['distance'])==0)
        total+=len(bad);useful+=int((c['out_error']<c['base_error']-1e-9).sum())
        harmful+=int(bad.sum());hard+=int(difficult.sum())
    return dict(query_observations=total,useful=useful,protected_harmful=harmful,
        high_similarity_zero_hamming_wrong_harmful=hard,
        diagnostic_rule='candidate damage > 1e-9, protected, wrong identity, top cosine >= .99, min Hamming = 0; labels never enter inference')


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');wp=Path('runs/v14_reference/post_gates.pt');ch=wp.with_name('choices.json')
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,ch]}
    projections=torch.load(cp,map_location='cpu');old_weights=torch.load(wp,map_location='cpu');old_choices=json.loads(ch.read_text())
    names=['dense_random','rare_details','correlated_dense','repeated_onehot'];mode='raw_int8'
    selected=[];logs=[];weights={}
    with torch.no_grad():
        for init in [41001,41002,41003]:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(projections[str(('positive_only',init))]);model.requires_grad_(False)
            def make(n,s):return cache(projected(mixed(n,s,mode),n,s,'learned_context',model))
            training={(n,s):make(n,s) for n in names[:2] for s in range(2600,2728)}
            cal=[make(n,s) for n in names for s in range(2740,2756)]
            val=[make(n,s) for n in names for s in range(2800,2900)]
            arms={}
            for size in [16,128]:
                cs=[training[n,s] for n in names[:2] for s in range(2600,2600+size)]
                gates,trace=train_path(cs,init,steps=(1000,));gate=gates[1000]
                choice,grid=select(cal,val,gate);arm='train_'+str(size);arms[arm]=(gate,choice)
                logs.append(dict(initialization=init,arm=arm,context_seeds=size,selected=choice,grid=grid,
                    trace=trace,coverage=coverage(cs),steps=1000,batch_size=512,learning_rate=.01))
                weights[str((init,arm))]=gate.state_dict()
                print(json.dumps(dict(initialization=init,arm=arm,selected=choice)),flush=True)
            frozen=UtilityGate();frozen.load_state_dict(old_weights[str((mode,init,1000))]);frozen.requires_grad_(False)
            matches=[c for c in old_choices if c['mode']==mode and c['initialization']==init and c['step']==1000];assert len(matches)==1
            choice=matches[0]['selected'];arms['frozen_v14']=(frozen,choice)
            logs.append(dict(initialization=init,arm='frozen_v14',selected=choice,frozen_weights_and_threshold=True))
            weights[str((init,'frozen_v14'))]=frozen.state_dict();selected.append((init,model,arms))
            del training,cal,val,cs
        # Freeze and save all arms before constructing the fresh test set.
        torch.save(weights,out/'post_gates.pt')
        (out/'choices.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
        count=0
        with (out/'comparisons.jsonl').open('w',encoding='utf-8') as f:
            for init,model,arms in selected:
                for n in names:
                    for seed in range(3000,3100):
                        c=cache(projected(mixed(n,seed,mode),n,seed,'learned_context',model))
                        for arm,(gate,choice) in arms.items():
                            row=evaluate(c,gate,choice);row.update(workload=n,seed=seed,mode=mode,initialization=init,arm=arm)
                            assert row['total_tensor_bytes']<=2048 and row['writes']<=25
                            f.write(json.dumps(row)+'\n');count+=1
                    f.flush();print(json.dumps(dict(initialization=init,workload=n,rows=count)),flush=True)
    assert count==3600
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,seconds=time.perf_counter()-start,
        provenance=provenance,training=list(range(2600,2728)),small_training=list(range(2600,2616)),
        calibration=list(range(2740,2756)),validation=list(range(2800,2900)),test=list(range(3000,3100)),
        production_ready=False,limitations=['Same 1000 optimizer updates and batch size, not equal wall time or total preprocessing cost.',
        '128 vs 16 seeds per active workload; smaller set nested in larger. Feature normalization refit per training arm.',
        'Fresh retrained arms select thresholds on shared new validation; frozen V14 retains old thresholds.',
        'Primary causal coverage comparison is train_128 vs train_16; frozen control differs in selection data.',
        'Same 100 test contexts reused across three initializations, not independent training replications.',
        'Frozen synthetic banks and projections; no full-model FLOP or language-model training claim.']),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
