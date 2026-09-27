import argparse,json,statistics
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root);rs=[json.loads(s) for s in (root/'comparisons.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rs)==3600;gs=defaultdict(list);lookup={}
    for r in rs:
        assert r['mode']=='raw_int8' and r['attempted_reads']<=64 and r['writes']<=25 and r['total_tensor_bytes']<=2048
        assert r['arm'] in ['train_16','train_128','frozen_v14']
        key=(r['workload'],r['initialization'],r['arm']);full=key+(r['seed'],)
        assert full not in lookup;lookup[full]=r;gs[key].append(r)
    assert len(gs)==36;groups=[];paired=[]
    for key,rows in gs.items():
        assert sorted(r['seed'] for r in rows)==list(range(3000,3100))
        g=dict(zip(['workload','initialization','arm'],key),preserved_seeds=sum(r['subgroup_preservation_pass'] for r in rows),
            failed_seeds=[r['seed'] for r in rows if not r['subgroup_preservation_pass']],
            protected_harmful_acceptances_total=sum(r['protected_harmful_acceptances'] for r in rows))
        for n in ['noisy_gain_percent','attempted_reads','applied_corrections','protected_harm_sum','mse','full_key_comparisons','post_linear_ops']:
            g[n]=statistics.mean(r[n] for r in rows)
        groups.append(g)
        if key[-1]=='train_128':
            for control in ['train_16','frozen_v14']:
                bs=[lookup[key[:2]+(control,r['seed'])] for r in rows]
                paired.append(dict(zip(['workload','initialization'],key[:2]),control=control,
                    wins=sum(r['noisy_mse']<b['noisy_mse']-1e-9 for r,b in zip(rows,bs)),
                    losses=sum(r['noisy_mse']>b['noisy_mse']+1e-9 for r,b in zip(rows,bs)),
                    mean_noisy_mse_change=statistics.mean(r['noisy_mse']-b['noisy_mse'] for r,b in zip(rows,bs))))
    result=dict(rows=len(rs),resource_caps_pass=True,groups=groups,paired=paired,
        general_preservation={arm:all(g['preserved_seeds']==100 for g in groups if g['arm']==arm) for arm in ['train_16','train_128','frozen_v14']})
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
