"""Reproduce the decision statistics from downloaded, immutable v2 evidence."""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root)
    main=[json.loads(s) for s in (root/'runs/v2_reference/comparisons.jsonl').read_text().splitlines()]
    extra=[json.loads(s) for s in (root/'runs/v2_extra_reference/comparisons.jsonl').read_text().splitlines()]
    assert len(main)==1920 and len(extra)==960
    groups=defaultdict(list)
    seen=set()
    for row in main:
        key=tuple(row[k] for k in ('workload','mode','write_budget','read_budget'))
        assert (*key,row['seed']) not in seen
        seen.add((*key,row['seed']))
        assert row['write_rate']<=row['write_budget']+1e-9
        assert row['retrieval_rate']<=row['read_budget']+1e-9
        assert row['peak_tensor_bytes']<=row['byte_budget']
        groups[key].append(row)
    improvements=[]
    for key,rows in groups.items():
        assert {r['seed'] for r in rows}=={100,101,102,103,104}
        change=[(r['mse']-r['baseline_mse'])/r['baseline_mse'] for r in rows]
        improvements.append(dict(setting=key,mean_relative_change=st.mean(change),
                                 better_seeds=sum(v<0 for v in change),seed_changes=change))
    extra_seen=set()
    for r in extra:
        key=tuple(r[k] for k in ('workload','mode','byte_budget','read_budget','topk','seed'))
        assert key not in extra_seen
        extra_seen.add(key)
        assert r['write_rate']<=r['write_budget']+1e-9
        assert r['retrieval_rate']<=r['read_budget']+1e-9
        assert r['peak_tensor_bytes']<=r['byte_budget']
    return dict(main_runs=len(main),extra_runs=len(extra),total_runs=len(main)+len(extra),
        matched_seeds=[100,101,102,103,104],main_settings=len(groups),all_resource_caps_pass=True,
        settings_with_mean_gain_over_0p1_percent=sum(r['mean_relative_change']<-.001 for r in improvements),
        exploratory_best=min(improvements,key=lambda r:r['mean_relative_change']),
        caveat='Exploratory summary across shared test seeds, not an independent significance test or a new tuning set.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',default='reports/evidence/v2')
    parser.add_argument('--out',default='reports/v2_decision_metrics.json')
    args=parser.parse_args();result=analyze(args.evidence)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
