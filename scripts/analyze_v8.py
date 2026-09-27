import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root);rows=[json.loads(x) for x in (root/'comparisons.jsonl').read_text().splitlines()]
    assert len(rows)==2880;grouped=defaultdict(list)
    for r in rows:
        assert r['queries']==1280 and r['attempted_reads']<=64 and r['writes']<=25 and r['total_tensor_bytes']<=2048
        assert sum(r['stage_counts'].values())==1280
        assert r['stage_counts']['applied']==r['applied_corrections']
        assert r['stage_counts']['applied']+r['stage_counts']['post_reject']==r['attempted_reads']
        assert sum(r['useful_stored_by_stage'].values())==r['useful_stored_total']
        grouped[r['workload'],r['mode'],r['projection'],r['initialization'],r['arm']].append(r)
    result=[];paired=[]
    for key,rs in sorted(grouped.items()):
        assert sorted(r['seed'] for r in rs)==list(range(1300,1320))
        g=dict(zip(['workload','mode','projection','initialization','arm'],key))
        for k in rs[0]:
            if k in ['seed','initialization','subgroup_preservation_pass']:continue
            vals=[r[k] for r in rs if isinstance(r[k],(int,float))]
            if vals:g[k]=statistics.mean(vals)
        for field in ['stage_counts','useful_stored_by_stage','useful_stored_mse_gain_sum_by_stage']:
            g[field]={s:statistics.mean(r[field][s] for r in rs) for s in rs[0][field]}
        g['preserved_seeds']=sum(r['subgroup_preservation_pass'] for r in rs)
        for prefix in ['', 'clean_', 'noisy_']:
            gains=[100*(r[prefix+'baseline_mse']-r[prefix+'mse'])/max(r[prefix+'baseline_mse'],1e-12) for r in rs]
            g[prefix+'gain_percent']=statistics.mean(gains);g[prefix+'positive_seeds']=sum(x>1e-7 for x in gains)
            g[prefix+'negative_seeds']=sum(x< -1e-7 for x in gains)
        result.append(g)
        if key[-1]=='shared_utility':
            for arm in ['shared_similarity','task_utility']:
                control={r['seed']:r for r in grouped[key[:-1]+(arm,)]}
                paired.append(dict(zip(['workload','mode','projection','initialization'],key[:-1]),control=arm,
                    noisy_wins=sum(r['noisy_mse']<control[r['seed']]['noisy_mse']-1e-9 for r in rs),
                    noisy_losses=sum(r['noisy_mse']>control[r['seed']]['noisy_mse']+1e-9 for r in rs),
                    shared_preserved=g['preserved_seeds'],control_preserved=sum(r['subgroup_preservation_pass'] for r in control.values())))
    out=dict(rows=len(rows),resource_caps_pass=True,stage_accounting_pass=True,groups=result,paired=paired)
    (root/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
