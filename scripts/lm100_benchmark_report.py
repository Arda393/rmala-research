"""Collect completed, identity-matched benchmark rows without changing models."""
import argparse
import collections
import fcntl
import json
from pathlib import Path
import random
import statistics
from rmala.lm100_bench import sha

VARIANTS = ['gla','v14_full','v14_half','full','hola']


def atomic(path, content):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(content, encoding='utf-8')
    tmp.replace(path)


def main(args):
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)
    lock = (root/'report.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    results, examples = {}, {}
    for v in VARIANTS:
        path = root/v/'result.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        assert r['status']=='completed' and not r['identity']['smoke'] and r['examples']==2240
        results[v] = r
        examples[v] = {x['id']:x for x in map(json.loads,(root/v/'examples.jsonl').read_text().splitlines())}
    assert len({r['identity']['cases_sha256'] for r in results.values()}) <= 1
    assert len({r['identity']['train_target_chain_sha256'] for r in results.values()}) <= 1
    paired = []
    if 'gla' in examples:
        for variant in examples:
            if variant == 'gla':
                continue
            assert examples[variant].keys() == examples['gla'].keys()
            groups = collections.defaultdict(list)
            for name, r in examples[variant].items():
                groups[(r['task'],r['length'],r['condition'])].append(r['correct']-examples['gla'][name]['correct'])
            for key, differences in sorted(groups.items()):
                rng = random.Random(20260927)
                boot = sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(2000))
                paired.append(dict(variant=variant,reference='gla',task=key[0],length=key[1],condition=key[2],
                    delta_accuracy=statistics.mean(differences),paired_bootstrap_ci95=[boot[49],boot[1949]],
                    scope='Per-cell example resampling only; one training seed; no multiple-comparison correction'))
    summary = dict(status='complete' if len(results)==5 else 'partial',completed=list(results),
                   results=results,paired_vs_gla=paired)
    atomic(root/'summary.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines = ['# Ortak yetenek ve FLOP testleri', '',
        'Tamamlanan modeller: '+', '.join(results), '',
        'Özgün Türkçe dört seçenekli tanı testidir; resmî RULER skoru değildir. Şans seviyesi %25.',
        '4096 eğitim uzunluğu dışındadır. Tablodaki uzunluk yürütme bütçesidir; gerçek prompt uzunluğu örnek kayıtlarında bulunur.',
        'Her hücre 64 örnek içerir. Tek eğitim seed’i kullanılmıştır; küçük farklar kesin üstünlük göstermez.', '',
        '| Görev | Token bütçesi | Koşul | '+' | '.join(VARIANTS)+' |',
        '|---|---:|---|'+'---:|'*5]
    cells = sorted({(c['task'],c['length'],c['condition']) for r in results.values() for c in r['cells']})
    for task,length,condition in cells:
        scores = []
        for v in VARIANTS:
            c = next((c for c in results.get(v,{}).get('cells',[]) if (c['task'],c['length'],c['condition'])==(task,length,condition)),None)
            scores.append(f"{c['accuracy']*100:.1f}%" if c else 'bekliyor')
        lines.append(f'| {task} | {length} | {condition} | '+' | '.join(scores)+' |')
    lines += ['', '## Inference hesap ve süre', '',
        'Batch=1, BF16, tüm konumlarda vocabulary logits üreten forward/prefill. KV-cache ile token üretme hızı değildir.',
        'FLOP algoritmik tahmindir; profiler kapsamı ve hariç tutulan işlemler JSON içinde ayrıca kayıtlıdır.', '',
        '| Model | Token | Tahmini FLOP/forward | Profiler kapsamındaki FLOP | Medyan süre (ms) | GPU tepe (GiB) |',
        '|---|---:|---:|---:|---:|---:|']
    for v,r in results.items():
        for c in r['compute']:
            lines.append(f"| {v} | {c['length']} | {c['estimate']['total_algorithmic_flops_estimate']:.4e} | {c['profiler_covered_flops']:.4e} | {c['median_seconds']*1000:.2f} | {c['peak_gpu_allocated_bytes']/1024**3:.2f} |")
    atomic(root/'REPORT.md','\n'.join(lines)+'\n')
    print(json.dumps(dict(status=summary['status'],completed=list(results),report=str(root/'REPORT.md'))),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='runs/lm100_benchmark_v2')
    main(p.parse_args())
