import argparse,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed,groups
from .experiments_v5 import projected
from .experiments_v7 import cache,probabilities
from .experiments_v13 import replay
from .nil_gate import match_targets
from .utility_gate import UtilityGate
from .v21_threshold_curve import sweep,summarize


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');wp=Path('runs/v14_reference/post_gates.pt');vp=Path('runs/v21_reference/post_gates.pt')
    bc=wp.with_name('choices.json');vc=vp.with_name('choices.json')
    states=torch.load(cp,map_location='cpu');baseweights=torch.load(wp,map_location='cpu');newweights=torch.load(vp,map_location='cpu')
    choices=json.loads(bc.read_text());newchoices=json.loads(vc.read_text());summaries=[]
    with torch.no_grad():
        for init in [41001,41002,41003]:
            model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))]);model.requires_grad_(False)
            choice=next(x['selected'] for x in choices if x['mode']=='raw_int8' and x['initialization']==init and x['step']==1000)
            gates={};events={};sizes={};baseline=queries=reads=0
            for arm in ['frozen_v14','utility_fine','nil_hard_fine']:
                g=UtilityGate();g.load_state_dict(baseweights[str(('raw_int8',init,1000))] if arm=='frozen_v14' else newweights[str((init,arm))]);g.requires_grad_(False)
                gates[arm]=g;events[arm]=[]
            for name in ['dense_random','rare_details']:
                for seed in range(4800,4900):
                    c=cache(projected(mixed(name,seed,'raw_int8'),name,seed,'learned_context',model))
                    prefix=f'{name}:{seed}';gm={prefix+'|'+n:m for n,m in groups(c)}
                    sizes.update({n:int(m.sum()) for n,m in gm.items() if m.any()})
                    baseline+=float(c['base_error'].double().sum());queries+=len(c['q'])
                    eligible,n,_=replay(c,[1.]*len(c['q']),0.,choice['hamming']);reads+=n
                    correct=match_targets(c['ids'],c['selected_ids'])
                    # Same float32 per-query delta used by safe_errors, accumulated in double.
                    delta=c['out_error']-c['base_error']
                    ids=eligible.nonzero().flatten().tolist()
                    for arm,g in gates.items():
                        ps=probabilities(g,c)
                        events[arm].extend(dict(score=ps[i],delta=float(delta[i]),correct=bool(correct[i]),missing=bool(~c['stored'][i]),
                            groups=[n for n,m in gm.items() if m[i]]) for i in ids)
                print(json.dumps(dict(initialization=init,workload=name,status='cached')),flush=True)
            for arm,es in events.items():
                rows=sweep(es,sizes,baseline,queries,reads)
                t=choice['threshold'] if arm=='frozen_v14' else next(x['selected']['threshold'] for x in newchoices if x['initialization']==init and x['arm']==arm)
                result=dict(initialization=init,arm=arm,queries=queries,contexts=200,hamming=choice['hamming'],breakpoints=len(rows),**summarize(rows,t))
                if arm=='frozen_v14':
                    t2=next(x['selected']['threshold'] for x in newchoices if x['initialization']==init and x['arm']=='frozen_fine')
                    result['v21_frozen_fine_current']=summarize(rows,t2)['current']
                (out/f'curve_{init}_{arm}.json').write_text(json.dumps(rows))
                (out/f'events_{init}_{arm}.json').write_text(json.dumps(dict(events=es,group_sizes=sizes,baseline_sum=baseline,queries=queries,attempts=reads)))
                summaries.append(result)
                print(json.dumps(result),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(status='completed',seconds=time.perf_counter()-start,results=summaries,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,vp,bc,vc]},
        interpretation='Exploratory exact sweep on reused V21 validation 4800-4899. No training, no new test, no deployed threshold, no relaxed safety criterion adopted. Empty-bank controls excluded; 200 active contexts per initialization. Empirical frontiers are not held-out guarantees.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
