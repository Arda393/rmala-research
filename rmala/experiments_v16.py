"""Diagnostic replay of seen V15 contexts; never a fresh quality test."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed
from .experiments_v5 import projected
from .experiments_v7 import cache,probabilities
from .experiments_v13 import replay,evaluate
from .utility_gate import UtilityGate


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    paths=[Path('runs/v6_reference/projections.pt'),Path('runs/v14_reference/post_gates.pt'),
        Path('runs/v15_reference/frozen_policies.json'),Path('runs/v15_reference/comparisons.jsonl')]
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    projections=torch.load(paths[0],map_location='cpu');weights=torch.load(paths[1],map_location='cpu')
    policies=json.loads(paths[2].read_text())['policies']
    old=[json.loads(s) for s in paths[3].read_text().splitlines()]
    old={(r['workload'],r['initialization'],r['seed']):r for r in old if r['step']==1000}
    count=accepted=0
    with torch.no_grad(),(out/'accepted_queries.jsonl').open('w',encoding='utf-8') as f:
        for init in [41001,41002,41003]:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(projections[str(('positive_only',init))])
            gate=UtilityGate();gate.load_state_dict(weights[str(('raw_int8',init,1000))]);gate.requires_grad_(False)
            choice=next(p['choice'] for p in policies if p['initialization']==init and p['step']==1000)
            for name in ['dense_random','rare_details']:
                for seed in range(2400,2500):
                    c=cache(projected(mixed(name,seed,'raw_int8'),name,seed,'learned_context',model))
                    live=evaluate(c,gate,choice);reference=old[name,init,seed]
                    for k,v in live.items():assert v==reference[k],(name,init,seed,k)
                    ps=probabilities(gate,c);mask,_,err=replay(c,ps,choice['threshold'],choice['hamming'])
                    for i in mask.nonzero().flatten().tolist():
                        b=float(c['base_error'][i]);v=float(err[i]);rare=bool(c['rare'][i]);stored=bool(c['stored'][i])
                        row=dict(workload=name,initialization=init,seed=seed,query=i,noisy=bool(c['noisy'][i]),
                            rare=rare,stored=stored,query_identity=int(c['ids'][i]),selected_identity=int(c['selected_ids'][i]),
                            correct_identity=bool(c['ids'][i]==c['selected_ids'][i]),base_error=b,candidate_error=v,
                            damage=v-b,protected=(not rare or not stored),harmful=v>b+1e-9,useful=v<b-1e-9,
                            probability=ps[i],threshold=choice['threshold'],probability_margin=ps[i]-choice['threshold'],
                            min_hamming=c['distance'][i],features=c['utility_features'][i].tolist())
                        f.write(json.dumps(row)+'\n');accepted+=1
                    count+=1
                f.flush();print(json.dumps(dict(initialization=init,workload=name,contexts=count,accepted=accepted)),flush=True)
    assert count==600
    (out/'summary.json').write_text(json.dumps(dict(status='completed',contexts=count,accepted_queries=accepted,
        exact_v15_replay=True,provenance=provenance,seconds=time.perf_counter()-start,
        limitations=['Seen V15 test data reused for diagnosis only; no new generalization evidence.',
        'All accepted queries in all 600 active-bank contexts; excludes unattempted/rejected candidates.',
        'Target, identity, rare/stored flags are diagnostic labels only, not permitted gate inputs.',
        'No threshold tuning or model training performed.']),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
