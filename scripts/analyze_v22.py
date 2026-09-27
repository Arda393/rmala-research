import argparse,json,statistics
from pathlib import Path
from collections import defaultdict


def analyze(root):
    root=Path(root);rows=[json.loads(s) for s in (root/'comparisons.jsonl').read_text().splitlines()]
    assert len(rows)==10800;gs=defaultdict(list);seen=set()
    for r in rows:
        assert r['mode']=='raw_int8' and r['attempted_reads']<=64 and r['writes']<=25 and r['total_tensor_bytes']<=2048
        assert (r['policy']=='baseline' and r['alpha']==0.) or (r['policy'] in ['fixed','selected'] and r['alpha'] in [1.,.5,.25,.1])
        k=(r['workload'],r['initialization'],r['policy'],r['alpha']);full=k+(r['seed'],)
        assert full not in seen;seen.add(full);gs[k].append(r)
    assert len(gs)==108;result=[]
    for k,rs in gs.items():
        assert sorted(r['seed'] for r in rs)==list(range(5400,5500))
        d=dict(zip(['workload','initialization','policy','alpha'],k),preserved_seeds=sum(r['subgroup_preservation_pass'] for r in rs))
        for n in ['useful_applications','harmful_acceptances','correct_identity_acceptances','wrong_identity_acceptances','harm_sum','benefit_sum','protected_harm_sum','protected_harmful_acceptances','attempted_reads','blend_scalar_ops']:
            d[n+'_total']=sum(r[n] for r in rs)
        for n in ['mse','baseline_mse','noisy_gain_percent']:d[n]=statistics.mean(r[n] for r in rs)
        d['worst_query_harm']=max(r['worst_query_harm'] for r in rs);result.append(d)
    # Fixed-threshold alpha arms must use exactly the same retrieval decisions.
    for w in ['dense_random','rare_details','correlated_dense','repeated_onehot']:
        for init in [41001,41002,41003]:
            ref=gs[(w,init,'fixed',1.)]
            for a in [.5,.25,.1]:
                for b,r in zip(ref,gs[(w,init,'fixed',a)]):
                    for n in ['attempted_reads','applied_corrections','correct_identity_acceptances','wrong_identity_acceptances','full_key_comparisons']:assert b[n]==r[n]
    out=dict(rows=len(rows),resource_caps_pass=True,fixed_routing_parity=True,groups=result)
    (root/'analysis.json').write_text(json.dumps(out,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
