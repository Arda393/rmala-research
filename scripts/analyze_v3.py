"""Reproducible standard-library aggregation; paired seed means, no p-value claims."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(root):
    root=Path(root)
    rows=[json.loads(s) for s in (root/'comparisons.jsonl').read_text().splitlines()]
    assert len(rows)==960, len(rows)
    keys=[(r['workload'],r['mode'],r['noisy'],r['arm'],r['seed']) for r in rows]
    assert len(set(keys))==len(keys)
    grouped=defaultdict(list)
    for r in rows:
        assert r['total_tensor_bytes']<=2048 and r['writes']<=25
        assert r['attempted_reads']<=32 and r['applied_corrections']<=r['attempted_reads']
        grouped[tuple(r[k] for k in ['workload','mode','noisy','arm'])].append(r)
    result=[]
    for key,rs in sorted(grouped.items()):
        assert sorted(r['seed'] for r in rs)==list(range(300,310))
        out=dict(zip(['workload','mode','noisy','arm'],key))
        for metric in ['mse','baseline_mse','attempted_reads','applied_corrections',
                       'false_accept_rate','false_reject_rate','useful_read_fraction',
                       'stored_mse','stored_baseline_mse','missing_mse','missing_baseline_mse',
                       'near_wrong_mse','near_wrong_baseline_mse','rare_mse','rare_baseline_mse',
                       'ordinary_mse','ordinary_baseline_mse','comparisons','prechecks',
                       'sketch_comparisons','precheck_key_elements','index_build_key_elements',
                       'total_tensor_bytes','capacity']:
            vals=[r[metric] for r in rs if r[metric] is not None]
            out[metric]=statistics.mean(vals) if vals else None
        gains=[100*(r['baseline_mse']-r['mse'])/max(r['baseline_mse'],1e-12) for r in rs]
        out.update(relative_gain_percent=statistics.mean(gains),min_gain_percent=min(gains),
                   max_gain_percent=max(gains),positive_seeds=sum(g>1e-7 for g in gains),
                   negative_seeds=sum(g< -1e-7 for g in gains),seed_gains_percent=gains)
        result.append(out)
    report=dict(rows=len(rows),groups=len(result),resource_caps_pass=True,groups_data=result)
    (root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
    print(json.dumps(analyze(a.root),indent=2))
