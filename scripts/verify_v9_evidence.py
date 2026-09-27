"""Verify V9 archive reconstruction, source provenance, and derived metrics."""
import hashlib
import json
from pathlib import Path
from rmala.counterfactual import replay
from scripts.analyze_v9 import analyze
from scripts.pack_v9_candidates import unpack


def verify():
    root=Path(__file__).resolve().parents[1];e=root/'reports/evidence/v9';out=e/'runs/v9_reference'
    unpack(e,'v9_reference')
    assert hashlib.sha256((out/'candidates.jsonl').read_text().encode()).hexdigest()=='2be9f65a9e7cf11c6989dd775207b085918e0b2e795e4855a571e060d631f2d8'
    old=json.loads((out/'analysis.json').read_text());assert analyze(out)==old
    manifest=json.loads((out/'source_manifest.json').read_text());newline_exceptions=[]
    for name,digest in manifest.items():
        s=(root/name).read_text()
        if hashlib.sha256(s.encode()).hexdigest()!=digest:
            assert hashlib.sha256((s+'\n').encode()).hexdigest()==digest,name
            newline_exceptions.append(name)
    v8=root/'reports/evidence/v8/runs/v8_reference'
    provenance=json.loads((out/'summary.json').read_text())['provenance']
    for name,digest in provenance.items():
        filename='projection_checkpoint.pt' if name.endswith('projections.pt') else Path(name).name
        assert hashlib.sha256((v8/filename).read_bytes()).hexdigest()==digest,name
    keys=('workload','mode','projection','initialization','seed')
    rows={tuple(r[k] for k in keys)+(r['arm'],):r for r in map(json.loads,(out/'comparisons.jsonl').read_text().splitlines())}
    for line in (out/'candidates.jsonl').open():
        c=json.loads(line)
        for arm in ['baseline_policy','double_budget','unlimited_budget','bypass_gate','unlimited_bypass','oracle_same_reads']:
            row=rows[tuple(c[k] for k in keys)+(arm,)]
            r=replay(c['base'],c['candidate'],c['probability'],c['distance'],c['threshold'],c['hamming'],row['rate'],row['gate'],c['populated'])
            safe=all(sum(e-b for e,b,m in zip(r['errors'],c['base'],mask) if m)/max(1,sum(mask))<=1e-9 for n,mask in c['groups'].items() if n!='noisy')
            assert safe==row['subgroup_preservation_pass']
            mask=c['groups']['noisy'];n=sum(mask)
            nm=sum(e for e,m in zip(r['errors'],mask) if m)/n;nb=sum(b for b,m in zip(c['base'],mask) if m)/n
            assert abs(nm-row['noisy_mse'])<1e-12 and abs(nb-row['noisy_baseline_mse'])<1e-12
            assert abs(100*(nb-nm)/max(nb,1e-12)-row['noisy_gain_percent'])<1e-12
    result=dict(rows=5760,contexts=960,local_reanalysis_equal=True,candidates_lossless=True,
                source_files_verified=len(manifest),extra_remote_final_newline=newline_exceptions,provenance_verified=True)
    (out/'local_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':verify()
