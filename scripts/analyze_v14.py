import argparse,json,statistics
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root);rows=[json.loads(s) for s in (root/'comparisons.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==1440;gs=defaultdict(list);lookup={}
    for r in rows:
        assert r['attempted_reads']<=64 and r['writes']<=25 and r['total_tensor_bytes']<=2048
        key=(r['workload'],r['mode'],r['initialization'],r['step']);gs[key].append(r)
        lookup[key+(r['seed'],)]=r
        if r['step']==r['chosen_step']:gs[key[:3]+('validation_selected',)].append(r)
    groups=[];paired=[]
    for key,rs in gs.items():
        assert sorted(r['seed'] for r in rs)==list(range(2300,2320))
        assert len(set(r['chosen_step'] for r in rs))==1
        g=dict(zip(['workload','mode','initialization','step'],key),chosen_step=rs[0]['chosen_step'],preserved_seeds=sum(r['subgroup_preservation_pass'] for r in rs))
        for n in ['noisy_gain_percent','attempted_reads','applied_corrections','protected_harmful_acceptances','protected_harm_sum','useful_applications','mse']:
            g[n]=statistics.mean(r[n] for r in rs)
        groups.append(g)
        if key[-1]=='validation_selected':
            baseline=[lookup[key[:3]+(300,r['seed'])] for r in rs]
            paired.append(dict(zip(['workload','mode','initialization'],key[:3]),chosen_step=g['chosen_step'],
                wins=sum(r['noisy_mse']<b['noisy_mse']-1e-9 for r,b in zip(rs,baseline)),
                losses=sum(r['noisy_mse']>b['noisy_mse']+1e-9 for r,b in zip(rs,baseline))))
    result=dict(rows=len(rows),selected_rows=480,resource_caps_pass=True,groups=groups,paired=paired)
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
