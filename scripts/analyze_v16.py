import argparse,json,statistics
from pathlib import Path


def analyze(root):
    root=Path(root);rs=[json.loads(s) for s in (root/'accepted_queries.jsonl').read_text(encoding='utf-8').splitlines()]
    summary=json.loads((root/'summary.json').read_text());assert len(rs)==summary['accepted_queries']
    assert len({(r['workload'],r['initialization'],r['seed'],r['query']) for r in rs})==len(rs)
    harmful=[r for r in rs if r['harmful'] and r['protected']];useful=[r for r in rs if r['useful']]
    def stats(xs):
        if not xs:return dict(count=0)
        d=dict(count=len(xs),wrong_identity=sum(not r['correct_identity'] for r in xs),missing=sum(not r['stored'] for r in xs),
            ordinary=sum(not r['rare'] for r in xs),noisy=sum(r['noisy'] for r in xs),damage_sum=sum(r['damage'] for r in xs))
        for key in ['probability','probability_margin','min_hamming']:
            vs=[r[key] for r in xs];d[key]=dict(min=min(vs),median=statistics.median(vs),max=max(vs))
        d['features']=[dict(min=min(r['features'][i] for r in xs),median=statistics.median(r['features'][i] for r in xs),max=max(r['features'][i] for r in xs)) for i in range(8)]
        return d
    result=dict(contexts=summary['contexts'],accepted=len(rs),protected_harm=stats(harmful),useful=stats(useful),
        harmful_records=harmful,diagnostic_only=True)
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');print(json.dumps(analyze(p.parse_args().root),indent=2))
