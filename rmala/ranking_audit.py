"""Frozen V10 calibration ranking diagnostics; never selects a deployable policy.

Metric helpers use only the standard library so their tie behavior can be checked
without the remote PyTorch environment. Ranking across an entire context is
retrospective and does not satisfy the online prefix budget.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path


def average_precision(scores, labels):
    """Threshold-group AP: tied scores enter the precision/recall curve together."""
    if len(scores) != len(labels):
        raise ValueError('Mismatched scores and labels')
    positives = sum(labels)
    if not positives:
        return None
    buckets = {}
    for score, label in zip(scores, labels):
        count, positive = buckets.get(score, (0, 0))
        buckets[score] = count + 1, positive + int(label)
    seen = true = 0
    ap = 0.0
    for score in sorted(buckets, reverse=True):
        count, positive = buckets[score]
        seen += count
        true += positive
        ap += (positive / positives) * (true / seen)
    return ap


def fractional_topk(scores, k):
    """Expected selection weights under a uniformly random cutoff tie break."""
    if not 0 <= k <= len(scores):
        raise ValueError('Invalid k')
    if not k:
        return [0.0] * len(scores)
    cutoff = sorted(scores, reverse=True)[k - 1]
    above = sum(score > cutoff for score in scores)
    tied = sum(score == cutoff for score in scores)
    fraction = (k - above) / tied
    return [1.0 if score > cutoff else fraction if score == cutoff else 0.0
            for score in scores]


def ranking_metrics(scores, labels, signed_gain, rate=.05):
    """Targets/gains are evaluation-only; scores contain no oracle information."""
    if not scores or len(scores) != len(labels) or len(scores) != len(signed_gain):
        raise ValueError('Expected nonempty aligned inputs')
    if not all(math.isfinite(float(v)) for v in scores + signed_gain):
        raise ValueError('Non-finite metric input')
    n = len(scores)
    k = math.floor(rate * n + 1e-9)
    weights = fractional_topk(scores, k)
    positives = sum(labels)
    positive_gain = [g if label else 0.0 for label, g in zip(labels, signed_gain)]
    total_gain = sum(positive_gain)
    true = sum(w * int(y) for w, y in zip(weights, labels))
    captured = sum(w * g for w, g in zip(weights, positive_gain))
    net = sum(w * g for w, g in zip(weights, signed_gain))
    oracle = sum(sorted(positive_gain, reverse=True)[:k])
    return dict(n=n, k=k, positives=positives, positive_rate=positives / n,
                average_precision=average_precision(scores, labels),
                top5_precision=true / k if k else None,
                top5_recall=true / positives if positives else None,
                positive_gain_total=total_gain,
                top5_positive_gain_sum=captured,
                top5_positive_gain_capture=captured / total_gain if total_gain else None,
                top5_net_gain_sum=net,
                top5_net_gain_per_query=net / n,
                retrospective_oracle_top5_positive_gain_sum=oracle,
                top5_fraction_of_retrospective_oracle=captured / oracle if oracle else None,
                uniform_random_precision=positives / n,
                uniform_random_recall=k / n if positives else None,
                uniform_random_positive_gain_capture=k / n if total_gain else None,
                uniform_random_net_gain_sum=k / n * sum(signed_gain),
                unique_scores=len(set(scores)))


def quantiles(values):
    values = sorted(values)
    result = {}
    for q in (0, .5, .75, .9, .95, .99, 1):
        rank = q * (len(values) - 1)
        lo, hi = math.floor(rank), math.ceil(rank)
        result[str(q)] = values[lo] + (values[hi] - values[lo]) * (rank - lo)
    return result


def metric_self_check():
    """Small exact cases cover ties, perfect/reversed rankings, and zero signal."""
    assert average_precision([1, 1, 1, 1], [1, 0, 0, 0]) == .25
    assert average_precision([4, 3, 2, 1], [1, 1, 0, 0]) == 1.0
    assert abs(average_precision([4, 3, 2, 1], [0, 0, 1, 1]) - 5 / 12) < 1e-12
    assert fractional_topk([2, 1, 1, 0], 2) == [1.0, .5, .5, 0.0]
    assert fractional_topk([1, 1], 0) == [0.0, 0.0]
    metric = ranking_metrics([1] * 20, [1] + [0] * 19, [2, -4] + [0] * 18)
    assert abs(metric['top5_positive_gain_capture'] - .05) < 1e-12
    assert abs(metric['top5_net_gain_sum'] + .1) < 1e-12
    zero = ranking_metrics([1] * 20, [0] * 20, [0] * 20)
    assert zero['average_precision'] is None and zero['top5_recall'] is None
    assert zero['top5_positive_gain_capture'] is None
    return True


def run_audit(out, projection_path='runs/v6_reference/projections.pt',
              post_path='runs/v8_reference/utility_gates.pt',
              post_choices_path='runs/v8_reference/choices.json',
              pre_path='runs/v10_reference/pre_gates.pt',
              pre_choices_path='runs/v10_reference/choices.json'):
    import torch
    from torch import nn
    from .utility_gate import UtilityGate
    from .experiments_v4 import mixed
    from .experiments_v5 import projected
    from .experiments_v10 import prepare, probs

    metric_self_check()
    torch.set_num_threads(2)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path, summary_path = out / 'ranking_rows.jsonl', out / 'ranking_summary.json'
    if rows_path.exists() or summary_path.exists():
        raise FileExistsError('Ranking audit output already exists')
    start = time.perf_counter()
    paths = [Path(p) for p in (projection_path, post_path, post_choices_path,
                              pre_path, pre_choices_path)]
    projections = torch.load(paths[0], map_location='cpu')
    posts = torch.load(paths[1], map_location='cpu')
    post_choices = json.loads(paths[2].read_text())
    pres = torch.load(paths[3], map_location='cpu')
    pre_choices = json.loads(paths[4].read_text())
    seeds = list(range(1410, 1418))
    names = ['dense_random', 'rare_details']
    rows = []
    with torch.no_grad(), rows_path.open('x') as output:
        for mode in ['raw_fp32', 'raw_int8']:
            bases = {(n, s): mixed(n, s, mode) for n in names for s in seeds}
            for init in [41001, 41002, 41003]:
                model = nn.Linear(32, 16, bias=False)
                model.load_state_dict(projections[str(('positive_only', init))])
                post = UtilityGate()
                post.load_state_dict(posts[str((mode, 'positive_only', init, 'shared'))])
                pre = UtilityGate()
                pre.load_state_dict(pres[str((mode, init))])
                model.eval(); post.eval(); pre.eval()
                choice = next(x['utility'] for x in post_choices if x['mode'] == mode
                              and x['projection'] == 'positive_only'
                              and x['initialization'] == init and x['scope'] == 'shared')
                old_pre = next(x for x in pre_choices if x['mode'] == mode
                               and x['initialization'] == init)
                for (name, seed), base in bases.items():
                    c = prepare(projected(copy.deepcopy(base), name, seed,
                                          'learned_context', model), post)
                    assert c['bank'].entries, 'Only populated contexts enter this diagnostic'
                    learned = probs(pre, c)
                    distances = list(c['distance'])
                    eligible = [d <= choice['hamming'] for d in distances]
                    post_ok = [p >= choice['threshold'] for p in c['post_prob']]
                    gains = (c['base_error'] - c['out_error']).tolist()
                    signed = [g if h and a else 0.0 for g, h, a in zip(gains, eligible, post_ok)]
                    labels = [h and a and g > 1e-9 for g, h, a in zip(gains, eligible, post_ok)]
                    for method, scores in [('learned_pre', learned),
                                           ('min_hamming', [-float(d) for d in distances])]:
                        row = dict(mode=mode, initialization=init, workload=name,
                                   seed=seed, method=method, **ranking_metrics(scores, labels, signed))
                        row['score_quantiles'] = quantiles(scores)
                        row['hamming_eligible'] = sum(eligible)
                        row['frozen_post_threshold'] = choice['threshold']
                        row['frozen_hamming'] = choice['hamming']
                        if method == 'learned_pre':
                            row['v10_selected_pre_threshold'] = old_pre['selected']['threshold']
                            # Coverage only: no threshold is selected or validated here.
                            row['v10_grid_active_pass_counts'] = [
                                dict(threshold=r['threshold'],
                                     passing=sum(p >= r['threshold'] for p in learned),
                                     hamming_eligible_passing=sum(
                                         p >= r['threshold'] and e for p, e in zip(learned, eligible)))
                                for r in old_pre['grid']]
                        rows.append(row)
                        output.write(json.dumps(row) + '\n')
                output.flush()
                print(json.dumps(dict(ranking_audit=mode, initialization=init,
                                      rows=len(rows))), flush=True)
    group_fields = ['mode', 'initialization', 'workload', 'method']
    metrics = ['positive_rate', 'average_precision', 'top5_precision', 'top5_recall',
               'top5_positive_gain_capture', 'top5_net_gain_sum', 'top5_net_gain_per_query',
               'top5_fraction_of_retrospective_oracle', 'uniform_random_precision']
    groups = []
    for key in sorted({tuple(r[k] for k in group_fields) for r in rows}):
        selected = [r for r in rows if tuple(r[k] for k in group_fields) == key]
        group = dict(zip(group_fields, key))
        group.update(contexts=len(selected), contexts_with_positive_signal=sum(r['positives'] > 0 for r in selected))
        for metric in metrics:
            values = [r[metric] for r in selected if r[metric] is not None]
            group[metric + '_mean'] = sum(values) / len(values) if values else None
            group[metric + '_defined_contexts'] = len(values)
        groups.append(group)
    summary = dict(status='completed', rows=len(rows), contexts=len(rows) // 2,
                   calibration_seeds=seeds, seconds=time.perf_counter() - start,
                   provenance={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                   groups=groups, metric_self_check=True, production_ready=False,
                   limitations=[
                       'In-sample diagnostic on V10 training/calibration contexts; no independent generalization claim.',
                       'No training, new threshold selection, validation, or holdout use in this audit.',
                       'Positive label requires frozen Hamming/post acceptance and candidate MSE gain > 1e-9.',
                       'Ranking uses only frozen pre probabilities or observable min-Hamming distance; labels/gains are evaluation-only.',
                       'Top5 is a retrospective whole-context selection of floor(0.05*N) queries, not an online prefix-feasible router.',
                       'Top5 cutoff ties receive fractional weights, the expectation under uniform random tie breaking.',
                       'Average precision groups all equal scores at the same threshold; zero-positive AP/recall/gain capture are null.',
                       'Oracle top5 uses positive target gains and foreknowledge solely as an offline upper bound.',
                       'Net gain includes harmful post-accepted candidates; positive capture alone does not establish safety.',
                       'Only the two populated-bank workloads are included; no empirical runtime or model-FLOP benefit is established.'])
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run_audit(args.out)
