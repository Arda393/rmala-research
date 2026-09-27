"""V8 frozen-policy diagnostic, on already observed V8 contexts. No training."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import torch
from torch import nn
from .counterfactual import replay
from .experiments_v4 import mixed,groups
from .experiments_v5 import projected
from .experiments_v7 import cache,probabilities
from .utility_gate import UtilityGate

ARMS=[('baseline_policy',.05,'frozen'),('double_budget',.1,'frozen'),
      ('unlimited_budget',1.,'frozen'),('bypass_gate',.05,'bypass'),
      ('unlimited_bypass',1.,'bypass'),('oracle_same_reads',.05,'oracle')]


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    began=time.perf_counter();root=Path('runs/v8_reference')
    cp=Path('runs/v6_reference/projections.pt');wp=root/'utility_gates.pt'
    states=torch.load(cp,map_location='cpu');weights=torch.load(wp,map_location='cpu')
    choices=[x for x in json.loads((root/'choices.json').read_text()) if x['scope']=='shared']
    previous={tuple(x[k] for k in ('workload','mode','projection','initialization','seed')):x
              for x in map(json.loads,(root/'comparisons.jsonl').read_text().splitlines()) if x['arm']=='shared_utility'}
    count=0;contexts=0
    with torch.no_grad(),(out/'comparisons.jsonl').open('w') as f,(out/'candidates.jsonl').open('w') as cf:
        for mode in ['raw_fp32','raw_int8']:
            bases={(n,s):mixed(n,s,mode) for n in ['dense_random','rare_details','correlated_dense','repeated_onehot'] for s in range(1300,1320)}
            for choice in [x for x in choices if x['mode']==mode]:
                proj=choice['projection'];init=choice['initialization'];policy=choice['utility']
                model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str((proj,init))])
                gate=UtilityGate();gate.load_state_dict(weights[str((mode,proj,init,'shared'))])
                for (name,seed),b in bases.items():
                    c=cache(projected(copy.deepcopy(b),name,seed,'learned_context',model))
                    info=dict(workload=name,mode=mode,projection=proj,initialization=init,seed=seed)
                    base=c['base_error'].tolist();cand=c['out_error'].tolist();prob=probabilities(gate,c)
                    masks={n:m.tolist() for n,m in groups(c)};masks['noisy']=c['noisy'].tolist()
                    evidence=dict(**info,base=base,candidate=cand,probability=prob,distance=c['distance'],
                                  threshold=policy['threshold'],hamming=policy['hamming'],populated=bool(c['bank'].entries),groups=masks)
                    cf.write(json.dumps(evidence)+'\n')
                    for arm,rate,action in ARMS:
                        r=replay(base,cand,prob,c['distance'],policy['threshold'],policy['hamming'],rate,action,bool(c['bank'].entries))
                        err=r.pop('errors');r.pop('mask');mse=sum(err)/len(err)
                        safe=all(sum(e-b for e,b,m in zip(err,base,mask) if m)/max(1,sum(mask))<=1e-9
                                 for n,mask in masks.items() if n!='noisy')
                        noisy=masks['noisy'];nb=sum(b for b,m in zip(base,noisy) if m)/sum(noisy)
                        nm=sum(e for e,m in zip(err,noisy) if m)/sum(noisy)
                        row=dict(**info,arm=arm,rate=rate,gate=action,**r,mse=mse,baseline_mse=sum(base)/len(base),
                                 noisy_mse=nm,noisy_baseline_mse=nb,noisy_gain_percent=100*(nb-nm)/max(nb,1e-12),
                                 subgroup_preservation_pass=safe,queries=len(base))
                        if arm=='baseline_policy':
                            old=previous[name,mode,proj,init,seed]
                            assert abs(mse-old['mse'])<1e-5
                            assert r['attempted_reads']==old['attempted_reads'] and r['applied_corrections']==old['applied_corrections']
                            assert safe==old['subgroup_preservation_pass']
                        f.write(json.dumps(row)+'\n');count+=1
                    contexts+=1
                print(json.dumps(dict(mode=mode,projection=proj,init=init,rows=count)),flush=True)
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,root/'choices.json',root/'comparisons.jsonl']}
    (out/'summary.json').write_text(json.dumps(dict(status='completed',rows=count,contexts=contexts,seconds=time.perf_counter()-began,
        baseline_v8_parity=True,provenance=provenance,diagnostic_only=True,fresh_holdout=False,
        limitations=['No training or deployable policy change. V8 test contexts reused for diagnosis.',
        'Interventions retain the frozen candidate and Hamming precheck; unlimited means read budget only.',
        '10% and 100% arms deliberately exceed the deployment read cap, only offline.',
        'Oracle uses target error after the same read schedule; not an implementable gate or global bank upper bound.',
        'No full-model FLOP or speed measurement.']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
