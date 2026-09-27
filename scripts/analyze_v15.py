import argparse,json,statistics
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root);rows=[json.loads(s) for s in (root/'comparisons.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==2400;gs=defaultdict(list);lookup={}
    for r in rows:
        assert r['mode']=='raw_int8'
        assert r['attempted_reads']<=64 and r['writes']<=25 and r['total_tensor_bytes']<=2048
        key=(r['workload'],r['initialization'],r['step']);full=key+(r['seed'],)
        assert full not in lookup;lookup[full]=r;gs[key].append(r)
    assert len(gs)==24;groups=[];paired=[]
    for key,rs in gs.items():
        assert sorted(r['seed'] for r in rs)==list(range(2400,2500))
        g=dict(zip(['workload','initialization','step'],key),preserved_seeds=sum(r['subgroup_preservation_pass'] for r in rs),
            failed_seeds=[r['seed'] for r in rs if not r['subgroup_preservation_pass']],
            protected_harmful_acceptances_total=sum(r['protected_harmful_acceptances'] for r in rs))
        for n in ['noisy_gain_percent','attempted_reads','applied_corrections','protected_harm_sum','mse','full_key_comparisons','post_linear_ops']:
            g[n]=statistics.mean(r[n] for r in rs)
        groups.append(g)
        if key[-1]==1000:
            bs=[lookup[key[:2]+(300,r['seed'])] for r in rs]
            paired.append(dict(zip(['workload','initialization'],key[:2]),
                wins=sum(r['noisy_mse']<b['noisy_mse']-1e-9 for r,b in zip(rs,bs)),
                losses=sum(r['noisy_mse']>b['noisy_mse']+1e-9 for r,b in zip(rs,bs)),
                mean_noisy_mse_change=statistics.mean(r['noisy_mse']-b['noisy_mse'] for r,b in zip(rs,bs))))
    result=dict(rows=len(rows),resource_caps_pass=True,groups=groups,paired=paired,
        candidate_general_preservation_pass=all(g['preserved_seeds']==100 for g in groups if g['step']==1000))
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
