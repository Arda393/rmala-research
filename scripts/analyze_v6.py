import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root);rows=[json.loads(x) for x in (root/'comparisons.jsonl').read_text().splitlines()]
    assert len(rows)==2400
    grouped=defaultdict(list)
    for r in rows:
        assert r['queries']==1280 and r['attempted_reads']<=64
        assert r['writes']<=25 and r['total_tensor_bytes']<=2048
        assert r['applied_corrections']<=r['attempted_reads']
        grouped[r['workload'],r['mode'],r['method'],r['initialization']].append(r)
    result=[]
    for key,rs in sorted(grouped.items()):
        assert sorted(r['seed'] for r in rs)==list(range(900,920))
        g=dict(zip(['workload','mode','method','initialization'],key))
        for k,v in rs[0].items():
            if k in ['seed','subgroup_preservation_pass']:continue
            vals=[r[k] for r in rs if isinstance(r[k],(int,float))]
            if vals:g[k]=statistics.mean(vals)
        g['preserved_seeds']=sum(r['subgroup_preservation_pass'] for r in rs)
        for prefix in ['', 'clean_', 'noisy_']:
            gains=[100*(r[prefix+'baseline_mse']-r[prefix+'mse'])/max(r[prefix+'baseline_mse'],1e-12) for r in rs]
            g[prefix+'gain_percent']=statistics.mean(gains)
            g[prefix+'positive_seeds']=sum(x>1e-7 for x in gains)
            g[prefix+'negative_seeds']=sum(x< -1e-7 for x in gains)
        result.append(g)
    paired=[]
    for (name,mode,method,init),rs in grouped.items():
        if method!='hard_negative':continue
        for control in ['positive_only','fixed_context']:
            lookup={r['seed']:r for r in grouped[name,mode,control,init]}
            pairs=[(r,lookup[r['seed']]) for r in rs]
            paired.append(dict(workload=name,mode=mode,initialization=init,control=control,
                noisy_mse_wins=sum(a['noisy_mse']<b['noisy_mse']-1e-9 for a,b in pairs),
                noisy_mse_losses=sum(a['noisy_mse']>b['noisy_mse']+1e-9 for a,b in pairs),
                noisy_mse_delta=statistics.mean(a['noisy_mse']-b['noisy_mse'] for a,b in pairs),
                hard_preserved=sum(a['subgroup_preservation_pass'] for a,b in pairs),
                control_preserved=sum(b['subgroup_preservation_pass'] for a,b in pairs)))
    output=dict(rows=len(rows),resource_caps_pass=True,groups=result,paired=paired)
    (root/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
