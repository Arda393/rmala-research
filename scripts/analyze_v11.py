import argparse,json,statistics
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root);fast=[json.loads(s) for s in (root/'fast/comparisons.jsonl').read_text().splitlines()]
    assert len(fast)==96 and all(r['outputs_equal'] for r in fast)
    assert all(r['dynamic_bytes']<=2048 and r['writes']<=25 and r['fast']['reads']<=64 for r in fast)
    rows=[json.loads(s) for s in (root/'threshold/comparisons.jsonl').read_text().splitlines()]
    assert len(rows)==1440;gs=defaultdict(list)
    for r in rows:
        assert r['attempted_reads']<=64 and r['writes']<=25 and r['total_tensor_bytes']<=2048
        gs[r['workload'],r['mode'],r['initialization'],r['arm']].append(r)
    groups=[];paired=[]
    for key,rs in sorted(gs.items()):
        assert sorted(r['seed'] for r in rs)==list(range(1700,1720))
        g=dict(zip(['workload','mode','initialization','arm'],key),preserved_seeds=sum(r['subgroup_preservation_pass'] for r in rs))
        for n in ['noisy_gain_percent','attempted_reads','applied_corrections','pre_evaluations','pre_linear_ops','mse']:
            g[n]=statistics.mean(r[n] for r in rs)
        groups.append(g)
        if key[-1]=='active_pool':
            for arm in ['mixed_pool','post_only']:
                old={r['seed']:r for r in gs[key[:-1]+(arm,)]}
                paired.append(dict(zip(['workload','mode','initialization'],key[:-1]),control=arm,
                    wins=sum(r['noisy_mse']<old[r['seed']]['noisy_mse']-1e-9 for r in rs),
                    losses=sum(r['noisy_mse']>old[r['seed']]['noisy_mse']+1e-9 for r in rs)))
    result=dict(threshold_rows=len(rows),fast_streams=len(fast),outputs_equal=True,resource_caps_pass=True,groups=groups,paired=paired,
                removed_pre_evaluations=sum(r['old']['pre_evaluations']-r['fast']['pre_evaluations'] for r in fast),
                removed_pre_linear_ops=sum(r['old']['pre_linear_ops']-r['fast']['pre_linear_ops'] for r in fast))
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
