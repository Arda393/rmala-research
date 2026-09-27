"""Output-equivalence audit of fixed-policy fast paths; no new learning."""
import argparse,copy,json,time,hashlib
from pathlib import Path
import torch
from torch import nn
from .experiments_v4 import mixed
from .experiments_v5 import projected
from .utility_gate import UtilityGate
from .pre_gate import PreRouter
from .fast_policy import FrozenFastRouter


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    cp=Path('runs/v6_reference/projections.pt');wp=Path('runs/v8_reference/utility_gates.pt')
    pp=Path('runs/v10_reference/pre_gates.pt');ch=Path('runs/v10_reference/choices.json')
    states=torch.load(cp,map_location='cpu');posts=torch.load(wp,map_location='cpu');pres=torch.load(pp,map_location='cpu')
    choices=json.loads(ch.read_text());rows=[]
    with torch.no_grad():
        for mode in ['raw_fp32','raw_int8']:
            bases={(n,s):mixed(n,s,mode) for n in ['dense_random','rare_details','correlated_dense','repeated_onehot'] for s in range(1500,1504)}
            for choice in [x for x in choices if x['mode']==mode]:
                init=choice['initialization'];model=nn.Linear(32,16,bias=False);model.load_state_dict(states[str(('positive_only',init))])
                post=UtilityGate();post.load_state_dict(posts[str((mode,'positive_only',init,'shared'))])
                pre=UtilityGate();pre.load_state_dict(pres[str((mode,init))])
                for (name,seed),b in bases.items():
                    c=projected(copy.deepcopy(b),name,seed,'learned_context',model)
                    args=(c['bank'],post,choice['post'],pre,choice['selected']['threshold'])
                    old=PreRouter(*args);new=FrozenFastRouter(*args)
                    for i,(q,base) in enumerate(zip(c['q'],c['base'])):
                        a=old.query(q,base);z=new.query(q,base)
                        assert torch.equal(a[0],z[0]) and a[1:]==z[1:]
                        assert new.attempts<=int(.05*(i+1)+1e-9)
                    row=dict(workload=name,mode=mode,initialization=init,seed=seed,queries=len(c['q']),outputs_equal=True,
                             dynamic_bytes=new.bytes,writes=c['bank'].write_budget.used)
                    for label,router in [('old',old),('fast',new)]:
                        row[label]=dict(reads=router.attempts,applied=router.applied,pre_evaluations=router.pre_evaluations,
                            pre_linear_ops=router.pre_evaluations*288,extra_sketch_comparisons=router.extra_sketch_comparisons,
                            post_evaluations=router.gate_evaluations,full_key_comparisons=router.comparisons)
                    assert new.bytes<=2048 and row['writes']<=25;rows.append(row)
                print(json.dumps(dict(mode=mode,init=init,streams=len(rows))),flush=True)
    (out/'comparisons.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=dict(status='completed',streams=len(rows),queries=sum(r['queries'] for r in rows),all_outputs_equal=True,
        seconds=time.perf_counter()-start,provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,wp,pp,ch]},
        removed_pre_evaluations=sum(r['old']['pre_evaluations']-r['fast']['pre_evaluations'] for r in rows),
        removed_pre_linear_ops=sum(r['old']['pre_linear_ops']-r['fast']['pre_linear_ops'] for r in rows),
        fresh_holdout=False,quality_gain_claim=False,full_model_speed_claim=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
