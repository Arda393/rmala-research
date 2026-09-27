import argparse,json,statistics
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root);rows=[json.loads(s) for s in (root/'comparisons.jsonl').read_text().splitlines()]
    assert len(rows)==1440;gs=defaultdict(list)
    for r in rows:
        assert r['attempted_reads']<=64 and r['writes']<=25 and r['total_tensor_bytes']<=2048
        if r['arm']=='cheap_sketch':assert r['pre_evaluations']==0 and r['pre_linear_ops']==0 and r['pre_required_static_bytes']==0
        gs[r['workload'],r['mode'],r['initialization'],r['arm']].append(r)
    groups=[];paired=[]
    for key,rs in sorted(gs.items()):
        assert sorted(r['seed'] for r in rs)==list(range(1900,1920))
        g=dict(zip(['workload','mode','initialization','arm'],key),preserved_seeds=sum(r['subgroup_preservation_pass'] for r in rs))
        for n in ['noisy_gain_percent','attempted_reads','applied_corrections','pre_evaluations','pre_linear_ops','mse',
                  'pre_required_static_bytes','sketch_comparisons','full_key_comparisons','post_linear_ops','harmful_acceptances']:
            g[n]=statistics.mean(r[n] for r in rs)
        groups.append(g)
        if key[-1]=='cheap_sketch':
            for arm in ['learned_pre','post_only']:
                old={r['seed']:r for r in gs[key[:-1]+(arm,)]}
                paired.append(dict(zip(['workload','mode','initialization'],key[:-1]),control=arm,
                    wins=sum(r['noisy_mse']<old[r['seed']]['noisy_mse']-1e-9 for r in rs),
                    losses=sum(r['noisy_mse']>old[r['seed']]['noisy_mse']+1e-9 for r in rs)))
    result=dict(rows=len(rows),resource_caps_pass=True,cheap_pre_cost_zero=True,groups=groups,paired=paired)
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
