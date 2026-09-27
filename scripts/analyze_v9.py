"""Standard-library audit and aggregation of offline V9 interventions."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from rmala.counterfactual import replay


def analyze(root):
    root=Path(root);rows=[json.loads(s) for s in (root/'comparisons.jsonl').read_text().splitlines()]
    assert len(rows)==5760
    keys=('workload','mode','projection','initialization','seed')
    bykey={tuple(r[k] for k in keys)+(r['arm'],):r for r in rows};assert len(bykey)==5760
    contexts=0
    for line in (root/'candidates.jsonl').open():
        c=json.loads(line);contexts+=1
        for arm in ['baseline_policy','double_budget','unlimited_budget','bypass_gate','unlimited_bypass','oracle_same_reads']:
            r=bykey[tuple(c[k] for k in keys)+(arm,)]
            audit=replay(c['base'],c['candidate'],c['probability'],c['distance'],c['threshold'],c['hamming'],r['rate'],r['gate'],c['populated'])
            assert r['attempted_reads']==audit['attempted_reads']
            for n in ['applied_corrections','helpful','harmful','benefit_sum','damage_sum']:assert r[n]==audit[n]
            assert abs(r['mse']-sum(audit['errors'])/len(audit['errors']))<1e-12
            assert r['attempted_reads']<=int(r['queries']*r['rate']+1e-9)
            if arm=='oracle_same_reads':assert r['harmful']==0
    assert contexts==960
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in keys[:-1])+(r['arm'],)].append(r)
    results=[]
    for key,rs in sorted(grouped.items()):
        assert sorted(r['seed'] for r in rs)==list(range(1300,1320))
        g=dict(zip(keys[:-1]+('arm',),key));g['preserved_seeds']=sum(r['subgroup_preservation_pass'] for r in rs)
        for n in ['mse','baseline_mse','noisy_gain_percent','attempted_reads','applied_corrections','helpful','harmful','benefit_sum','damage_sum']:
            g[n]=statistics.mean(r[n] for r in rs)
        controls=[bykey[tuple(r[k] for k in keys)+('baseline_policy',)] for r in rs]
        g['paired_noisy_wins']=sum(r['noisy_mse']<b['noisy_mse']-1e-9 for r,b in zip(rs,controls))
        g['paired_noisy_losses']=sum(r['noisy_mse']>b['noisy_mse']+1e-9 for r,b in zip(rs,controls))
        results.append(g)
    out=dict(rows=len(rows),contexts=contexts,replay_verified=True,diagnostic_only=True,groups=results)
    (root/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
