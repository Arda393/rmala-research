import argparse,json,statistics
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root);rows=[json.loads(s) for s in (root/'comparisons.jsonl').read_text().splitlines()]
    assert len(rows)==960;groups=defaultdict(list)
    for r in rows:
        assert r['attempted_reads']<=64 and r['writes']<=25 and r['total_tensor_bytes']<=2048
        groups[r['workload'],r['mode'],r['initialization'],r['arm']].append(r)
    results=[];paired=[]
    for key,rs in sorted(groups.items()):
        assert sorted(r['seed'] for r in rs)==list(range(1500,1520))
        g=dict(zip(['workload','mode','initialization','arm'],key))
        g['preserved_seeds']=sum(r['subgroup_preservation_pass'] for r in rs)
        for n in ['attempted_reads','applied_corrections','harmful_acceptances','mse','baseline_mse','pre_evaluations','pre_linear_arithmetic_ops','extra_sketch_comparisons']:
            g[n]=statistics.mean(r.get(n,0) for r in rs)
        gains=[100*(r['noisy_baseline_mse']-r['noisy_mse'])/max(r['noisy_baseline_mse'],1e-12) for r in rs]
        g['noisy_gain_percent']=statistics.mean(gains);g['noisy_positive_seeds']=sum(x>1e-7 for x in gains)
        results.append(g)
        if key[-1]=='pre_and_post':
            old={r['seed']:r for r in groups[key[:-1]+('frozen_post',)]}
            paired.append(dict(zip(['workload','mode','initialization'],key[:-1]),
                wins=sum(r['noisy_mse']<old[r['seed']]['noisy_mse']-1e-9 for r in rs),
                losses=sum(r['noisy_mse']>old[r['seed']]['noisy_mse']+1e-9 for r in rs)))
    a=dict(rows=len(rows),resource_caps_pass=True,groups=results,paired=paired)
    (root/'analysis.json').write_text(json.dumps(a,indent=2)+'\n');return a


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
