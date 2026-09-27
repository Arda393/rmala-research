import argparse,json,statistics
from pathlib import Path
from collections import defaultdict


def analyze(root):
    root=Path(root);rs=[json.loads(s) for s in (root/'comparisons.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rs)==2400;groups=defaultdict(list);lookup={}
    for r in rs:
        assert r['mode']=='raw_int8' and r['arm'] in ['frozen_v14','raw_verifier']
        assert r['attempted_reads']<=64 and r['writes']<=25 and r['total_tensor_bytes']<=2048
        key=(r['workload'],r['initialization'],r['arm']);full=key+(r['seed'],)
        assert full not in lookup;lookup[full]=r;groups[key].append(r)
    assert len(groups)==24;gs=[];paired=[]
    for key,rows in groups.items():
        assert sorted(r['seed'] for r in rows)==list(range(3800,3900))
        g=dict(zip(['workload','initialization','arm'],key),preserved_seeds=sum(r['subgroup_preservation_pass'] for r in rows),
            protected_harmful_acceptances_total=sum(r['protected_harmful_acceptances'] for r in rows),
            raw_hamming=rows[0]['raw_hamming'])
        for n in ['noisy_gain_percent','mse','attempted_reads','applied_corrections','protected_harm_sum','full_key_comparisons','post_linear_ops',
                  'verifier_checks','verifier_rejections','verifier_bytes','total_tensor_bytes','metadata_checks','raw_signature_elements']:
            g[n]=statistics.mean(r[n] for r in rows)
        gs.append(g)
        if key[-1]=='raw_verifier':
            bs=[lookup[key[:2]+('frozen_v14',r['seed'])] for r in rows]
            paired.append(dict(zip(['workload','initialization'],key[:2]),wins=sum(r['noisy_mse']<b['noisy_mse']-1e-9 for r,b in zip(rows,bs)),
                losses=sum(r['noisy_mse']>b['noisy_mse']+1e-9 for r,b in zip(rows,bs)),mean_noisy_mse_change=statistics.mean(r['noisy_mse']-b['noisy_mse'] for r,b in zip(rows,bs))))
    result=dict(rows=len(rs),resource_caps_pass=True,groups=gs,paired=paired,
        general_preservation={arm:all(g['preserved_seeds']==100 for g in gs if g['arm']==arm) for arm in ['frozen_v14','raw_verifier']})
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
