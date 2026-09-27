"""Run the identical locked diagnostics and warm inference FLOP samples on each LM."""
import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F
from rmala.lm100_bench import choice_rows, sha, summarize, oracle
from rmala.lm100_compute import collect_memory_stats, compute_estimate
from rmala.lm100_model import LM100, VARIANTS
from rmala.lm100_train import atomic_json, flop_profiler, vendor_revision


@torch.no_grad()
def score(model, record):
    xs, ys = choice_rows(record)
    x = torch.tensor(xs, device='cuda', dtype=torch.long)
    y = torch.tensor(ys, device='cuda', dtype=torch.long)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        losses = model.losses(x, y)
    values = (-losses.double().sum(dim=1)).cpu().tolist()
    assert len(values) == 4 and all(math.isfinite(v) for v in values)
    stats = collect_memory_stats(model)
    estimate = compute_estimate(model, 4, record['length'], stats, training=False)
    return values, estimate, {k:sum(m[k] for m in stats) for k in ['writes','reads','accepted','opportunities']}


@torch.no_grad()
def score_check(model, record):
    # Independent CE path verifies the answer mask, shift and fused scoring API.
    xs, ys = choice_rows(record)
    x = torch.tensor(xs, device='cuda', dtype=torch.long)
    y = torch.tensor(ys, device='cuda', dtype=torch.long)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x)
    ref = -F.cross_entropy(logits.float().transpose(1,2), y, reduction='none', ignore_index=-100).sum(1)
    del logits
    got, _, _ = score(model, record)
    torch.testing.assert_close(torch.tensor(got, device='cuda', dtype=ref.dtype), ref, atol=2e-4, rtol=2e-4)
    shuffled = dict(record, choice_ids=list(reversed(record['choice_ids'])))
    reversed_scores, _, _ = score(model, shuffled)
    assert max(abs(a-b) for a,b in zip(got, reversed(reversed_scores))) < .02


@torch.no_grad()
def compute_benchmark(model, records):
    results = []
    for length in [128,512,1024,2048,4096]:
        r = next(x for x in records if x['length'] == length)
        xs, _ = choice_rows(r)
        x = torch.tensor(xs[:1], device='cuda', dtype=torch.long)
        def forward():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
            del logits
        for _ in range(3):
            forward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(7):
            torch.cuda.synchronize()
            start = time.perf_counter()
            forward()
            torch.cuda.synchronize()
            durations.append(time.perf_counter()-start)
        memory = collect_memory_stats(model)
        estimate = compute_estimate(model, 1, length, memory, training=False)
        with flop_profiler() as profiler:
            forward()
            torch.cuda.synchronize()
        results.append(dict(length=length, batch_size=1, warmup=3, repetitions=7,
            median_seconds=statistics.median(durations), seconds=durations,
            input_tokens_per_second=length/statistics.median(durations),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            profiler_covered_flops=sum(e.flops for e in profiler.key_averages()),
            algorithmic_flops_per_input_token=estimate['total_algorithmic_flops_estimate']/length,
            estimate=estimate, scope='Full forward/prefill with all-position vocabulary logits; not cached generation latency'))
        print(json.dumps(dict(stage='compute', variant=model.variant, length=length, seconds=results[-1]['median_seconds'])), flush=True)
    return results


def main(args):
    suite = Path(args.suite)
    meta = json.loads((suite/'manifest.json').read_text())
    assert sha(suite/'cases.jsonl') == meta['cases_sha256']
    assert sha('rmala/lm100_bench.py') == meta['builder_sha256']
    records = [json.loads(x) for x in (suite/'cases.jsonl').read_text().splitlines()]
    assert len(records) == meta['examples']
    assert len({r['id'] for r in records}) == len(records)
    for r in records:
        assert r['options'][r['gold']] == oracle(r)
    cfg = json.loads(Path('configs/lm100_3b.json').read_text())
    checkpoint_path = Path(cfg['output_root'])/args.variant/'latest.pt'
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = saved['state']
    assert state['cursor'] == 3000000000 and state['status'] == 'completed'
    assert state['identity']['variant'] == args.variant
    assert state['identity']['config_sha256'] == sha('configs/lm100_3b.json')
    for name, expected in state['identity']['source_sha256'].items():
        assert sha(name) == expected, name
    assert vendor_revision() == state['identity']['hola_commit']
    out = Path(args.output)/args.variant
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'process.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(variant=args.variant, checkpoint_sha256=sha(checkpoint_path),
        checkpoint_identity=state['identity'], train_target_chain_sha256=state['target_chain_sha256'],
        cases_sha256=meta['cases_sha256'], runner_sha256=sha(__file__), smoke=args.smoke)
    identity_path = out/'identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, 'Different existing benchmark'
    else:
        atomic_json(identity_path, identity)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = LM100(args.variant, vocab_size=cfg['vocab_size'], dim=cfg['dim'], layers=cfg['layers'],
                  heads=cfg['heads'], seed=cfg['seed']).cuda()
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval()
    score_check(model, records[0])
    print(json.dumps(dict(stage='score_check_pass', variant=args.variant)), flush=True)
    compute_path = out/'compute.json'
    if compute_path.exists():
        compute = json.loads(compute_path.read_text())
    else:
        compute = compute_benchmark(model, records)
        atomic_json(compute_path, compute)
    if args.smoke:
        records = [r for r in records if r['instance'] == 0]
    results_path = out/'examples.jsonl'
    previous = []
    if results_path.exists():
        lines = results_path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                previous.append(json.loads(line))
            except json.JSONDecodeError:
                assert i == len(lines)-1, 'Corrupt nonterminal row'
        assert [r['id'] for r in previous] == [r['id'] for r in records[:len(previous)]]
        tmp = results_path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r)+'\n' for r in previous))
        os.replace(tmp, results_path)
    rows = list(previous)
    start = time.perf_counter()
    with results_path.open('a', buffering=1) as handle:
        for record in records[len(previous):]:
            logprobs, estimate, memory = score(model, record)
            prediction = max(range(4), key=lambda i: logprobs[i])
            ordered = sorted(logprobs, reverse=True)
            row = {k:record[k] for k in ['id','pair_id','task','length','condition','instance','regime',
                                       'gold','actual_prompt_tokens','answer_distance_tokens']}
            row.update(prediction=prediction, correct=int(prediction == record['gold']),
                       logprobs=logprobs, tie=ordered[0]-ordered[1] <= 1e-7,
                       scoring_algorithmic_flops=estimate['total_algorithmic_flops_estimate'], memory=memory)
            handle.write(json.dumps(row)+'\n')
            rows.append(row)
            if len(rows) % 128 == 0:
                atomic_json(out/'progress.json', dict(status='running', completed=len(rows), total=len(records)))
                print(json.dumps(dict(stage='scoring', variant=args.variant, completed=len(rows), total=len(records))), flush=True)
    result = dict(status='completed', identity=identity, examples=len(rows), chance_accuracy=.25,
        cells=summarize(rows), compute=compute, measured_scoring_seconds_this_process=time.perf_counter()-start,
        total_scoring_algorithmic_flops=sum(r['scoring_algorithmic_flops'] for r in rows),
        hardware=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda),
        note='Original Turkish forced-choice diagnostics, not official RULER; 4096 extrapolation is separate.')
    atomic_json(out/'result.json', result)
    atomic_json(out/'progress.json', dict(status='completed', completed=len(rows), total=len(records)))
    print(json.dumps(dict(status='PASS', variant=args.variant, examples=len(rows), output=str(out))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--suite', default='lm100/benchmark_v2')
    p.add_argument('--output', default='runs/lm100_benchmark_v2')
    p.add_argument('--smoke', action='store_true')
    main(p.parse_args())
