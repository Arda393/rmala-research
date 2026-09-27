import argparse,json,statistics
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root);rows=[json.loads(s) for s in (root/'comparisons.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==1440;gs=defaultdict(list)
    for r in rows:
        assert r['attempted_reads']<=64 and r['writes']<=25 and r['total_tensor_bytes']<=2048
        assert r['protected_harmful_acceptances']<=r['harmful_acceptances']<=r['applied_corrections']
        gs[r['workload'],r['mode'],r['initialization'],r['arm']].append(r)
    groups=[];paired=[]
    for key,rs in sorted(gs.items()):
        assert sorted(r['seed'] for r in rs)==list(range(2100,2120))
        g=dict(zip(['workload','mode','initialization','arm'],key),preserved_seeds=sum(r['subgroup_preservation_pass'] for r in rs))
        for n in ['noisy_gain_percent','attempted_reads','applied_corrections','harmful_acceptances','protected_harmful_acceptances',
                  'protected_harm_sum','useful_applications','sketch_comparisons','post_linear_ops','mse']:
            g[n]=statistics.mean(r[n] for r in rs)
        g['noisy_positive_seeds']=sum(r['noisy_gain_percent']>1e-7 for r in rs);groups.append(g)
        if key[-1]=='protected_harm4':
            for arm in ['retrained_standard','frozen_v8']:
                old={r['seed']:r for r in gs[key[:-1]+(arm,)]}
                paired.append(dict(zip(['workload','mode','initialization'],key[:-1]),control=arm,
                    noisy_wins=sum(r['noisy_mse']<old[r['seed']]['noisy_mse']-1e-9 for r in rs),
                    noisy_losses=sum(r['noisy_mse']>old[r['seed']]['noisy_mse']+1e-9 for r in rs),
                    lower_protected_harm=sum(r['protected_harm_sum']<old[r['seed']]['protected_harm_sum']-1e-9 for r in rs),
                    higher_protected_harm=sum(r['protected_harm_sum']>old[r['seed']]['protected_harm_sum']+1e-9 for r in rs)))
    result=dict(rows=len(rows),resource_caps_pass=True,groups=groups,paired=paired)
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
