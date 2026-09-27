"""Locked Turkish diagnostic tasks; not an official RULER benchmark score."""
import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path

VOCAB_SHA = '72412d981dac65a29d1767bc98821fc2bcffc2de53c534e7c719598515bfb600'
COLORS = ['mavi', 'sarı', 'mor', 'gri']
FILLERS = [
    'Sabah hava açıktı ve insanlar günlük işlerine devam ediyordu.\n',
    'Kütüphanede kitaplar düzenlendi ve masalar temizlendi.\n',
    'Yol boyunca ağaçlar vardı ve hafif bir rüzgar esiyordu.\n',
    'Toplantıda gelecek haftanın çalışma programı konuşuldu.\n',
    'Bahçedeki çiçekler sulandı ve pencereler açıldı.\n',
    'Gazetede şehir hayatı hakkında uzun bir yazı yayımlandı.\n',
    'Öğleden sonra herkes kendi çalışmasına geri döndü.\n',
    'Yeni defterler raflara yerleştirildi ve kapı kapatıldı.\n',
]


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def oracle(record):
    """Solve explicit facts independently of rendered answer labels."""
    facts = record['facts']
    if record['task'] == 'two_hop':
        aliases = {f['key']: f['value'] for f in facts if f['kind'] == 'alias'}
        colors = {f['key']: f['value'] for f in facts if f['kind'] == 'color'}
        answer = [colors[aliases[record['query_keys'][0]]]]
    else:
        colors = {f['key']: f['value'] for f in sorted(facts, key=lambda f: f['position'])}
        answer = [colors[k] for k in record['query_keys']]
    return ' ' + ', '.join(answer)


def spec(task, condition, index):
    color = COLORS[index % 4]
    other = [c for c in COLORS if c != color]
    key = str(3000 + index * 11)
    keys = [key, str(int(key)+1), str(int(key)+2)]
    facts = []
    def add(k, value, position, kind='color', update=False):
        facts.append(dict(key=k, value=value, position=position, kind=kind, update=update))
    options = [' ' + c for c in COLORS]
    query_keys = [key]
    if task == 'recall':
        add(key, color, float(condition))
    elif task == 'multi_retrieval':
        query_keys = keys
        tuples = [[COLORS[(j+k) % 4] for k in range(3)] for j in range(4)]
        truth = tuples[index % 4]
        candidates = [list(truth)]
        for j in range(3):
            foil = list(truth)
            foil[j] = COLORS[(COLORS.index(foil[j])+1) % 4]
            candidates.append(foil)
        # Every foil changes exactly one fact; any single retrieved key is insufficient.
        shift = index % 4
        candidates = candidates[-shift:] + candidates[:-shift] if shift else candidates
        options = [' ' + ', '.join(t) for t in candidates]
        for k, c, pos in zip(keys, tuples[index % 4], [.15, .45, .75]):
            add(k, c, pos)
    elif task == 'distractors':
        add(key, color, .5)
        if condition == 'similar_keys':
            for j, pos in enumerate([.1, .25, .4, .6, .75, .9]):
                add(str(int(key)+j+1), other[j % 3], pos)
    elif task == 'two_hop':
        positions = [.2, .7] if (index//4) % 2 == 0 else [.7, .2]
        add(key, keys[1], positions[0], kind='alias')
        add(keys[1], color, positions[1])
        for j, pos in enumerate([.1, .45, .9]):
            add(str(int(key)+j+3), other[j], pos)
    elif task == 'state_tracking':
        add(key, other[0], .1)
        add(key, other[1], .4, update=True)
        add(key, color, .75, update=True)
        add(keys[1], other[2], .9)
    else:
        raise ValueError(task)
    if task == 'multi_retrieval':
        question = '\nSonuç: ' + ', '.join(query_keys) + ' numaralı kutuların renkleri sırasıyla:'
    elif task == 'state_tracking':
        question = f'\nSonuç: {key} numaralı kutunun son rengi:'
    else:
        question = f'\nSonuç: {key} numaralı kutunun rengi:'
    return dict(task=task, condition=condition, instance=index, facts=facts,
                query_keys=query_keys, question=question, options=options)


def build_record(tokenizer, task, condition, index, length):
    r = spec(task, condition, index)
    r['length'] = length
    r['id'] = f'{task}/{length}/{condition}/{index:03d}'
    r['pair_id'] = f'{task}/{length}/{index:03d}'
    r['gold'] = r['options'].index(oracle(r))
    enc = lambda s: list(tokenizer.encode_ids(s.encode('utf-8')))
    choices = [enc(x) for x in r['options']]
    # Equal answer lengths prevent a shorter continuation winning by construction.
    assert len({len(x) for x in choices}) == 1, r['id']
    r['choice_ids'] = choices
    question = enc(r['question'])
    r['question_ids'] = [2] + question
    target_prompt = length - len(choices[0]) + 1
    prefix = enc('Notlar:\n')
    chunks = []
    for f in sorted(r['facts'], key=lambda f: f['position']):
        if f['kind'] == 'alias':
            text = f"{f['key']} numaralı kutu, {f['value']} numaralı kutuyla aynı renktedir.\n"
        elif f['update']:
            text = f"Daha sonra {f['key']} numaralı kutu yeniden boyandı. Yeni rengi {f['value']}.\n"
        else:
            text = f"{f['key']} numaralı kutunun rengi {f['value']}.\n"
        chunks.append((f, enc(text)))
    available = target_prompt - 1 - len(prefix) - len(question) - sum(len(x) for _, x in chunks)
    if available < 0:
        raise ValueError(f'Task does not fit context: {r["id"]}')
    rng = random.Random(20260927 + index)
    sentences = [enc(x) for x in FILLERS]
    newline = enc('\n')
    assert len(newline) == 1
    def filler(n):
        out = []
        while n >= max(map(len, sentences)):
            x = rng.choice(sentences)
            out.extend(x)
            n -= len(x)
        return out + newline*n
    ids = [2] + prefix
    used = 0
    for f, tokens in chunks:
        cumulative = int(available*f['position'])
        ids.extend(filler(cumulative-used))
        used = cumulative
        f['token_start'] = len(ids)
        ids.extend(tokens)
        f['token_end'] = len(ids)
    ids.extend(filler(available-used))
    ids.extend(question)
    assert len(ids) == target_prompt
    # Canonicalize the whole text: native tokenizer merges adjacent newlines.
    # Length is the execution budget; record actual prompt length and fact offsets.
    raw = tokenizer.decode_ids(ids[1:])
    original_ids = ids
    ids = [2] + list(tokenizer.encode_ids(raw))
    assert len(ids) <= target_prompt, r['id']
    for f in r['facts']:
        for key in ('token_start', 'token_end'):
            f[key] = 1 + len(tokenizer.encode_ids(tokenizer.decode_ids(original_ids[1:f[key]])))
    for text, tokens in zip(r['options'], choices):
        assert list(tokenizer.encode_ids(raw+text.encode('utf-8'))) == ids[1:]+tokens, r['id']
    r['prompt_ids'] = ids
    r['actual_prompt_tokens'] = len(ids)
    r['prompt_text'] = raw.decode('utf-8')
    r['regime'] = 'extrapolation' if length > 2048 else 'within_training_length'
    r['answer_distance_tokens'] = len(ids) - max(f['token_end'] for f in r['facts'] if f['key'] in r['query_keys'])
    return r


def cell_specs():
    for length in [128, 512, 1024, 2048, 4096]:
        for pos in ['0.1', '0.5', '0.9']:
            yield 'recall', pos, length
    for length in [512, 1024, 2048, 4096]:
        yield 'multi_retrieval', 'three_keys', length
        for condition in ['clean', 'similar_keys']:
            yield 'distractors', condition, length
        yield 'two_hop', 'two_facts', length
        yield 'state_tracking', 'three_states', length


def wilson(correct, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = correct/n
    mid = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., mid-half), min(1., mid+half)]


def choice_rows(record):
    """Teacher forcing: first answer target is predicted from the last prompt token."""
    prompt = record['prompt_ids']
    length = record['length']
    xs, ys = [], []
    for answer in record['choice_ids']:
        sequence = prompt + answer
        assert len(sequence)-1 <= length and prompt and answer
        x = sequence[:-1] + [2]*(length-len(sequence)+1)
        y = [-100]*length
        y[len(prompt)-1:len(prompt)+len(answer)-1] = answer
        xs.append(x)
        ys.append(y)
    return xs, ys


def summarize(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['task'], r['length'], r['condition'])].append(r)
    return [dict(task=k[0], length=k[1], condition=k[2], n=len(v),
                 accuracy=sum(x['correct'] for x in v)/len(v),
                 accuracy_ci95=wilson(sum(x['correct'] for x in v), len(v)),
                 mean_gold_logprob=sum(x['logprobs'][x['gold']] for x in v)/len(v),
                 ties=sum(x['tie'] for x in v)) for k, v in sorted(groups.items())]


def build(args):
    from nedotokenizer import SurfaceTokenizer
    assert sha(args.vocab) == VOCAB_SHA
    assert args.per_cell > 0 and args.per_cell % 4 == 0
    tokenizer = SurfaceTokenizer(Path(args.vocab).read_bytes())
    records = [build_record(tokenizer, task, condition, i, length)
               for task, condition, length in cell_specs() for i in range(args.per_cell)]
    payload = ''.join(canonical(r)+'\n' for r in records)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = out/'cases.jsonl'
    if data.exists() and data.read_text(encoding='utf-8') != payload:
        raise RuntimeError('Refusing to overwrite a different locked suite')
    data.write_text(payload, encoding='utf-8')
    meta = dict(schema='rmala_diagnostic_benchmark_v1', examples=len(records),
                per_cell=args.per_cell, cells=len(list(cell_specs())),
                cases_sha256=sha(data), vocab_sha256=VOCAB_SHA,
                builder_sha256=sha(__file__), chance_accuracy=.25,
                primary='4-choice summed conditional answer log likelihood; equal-length answers',
                provenance='Original Turkish synthetic diagnostic suite; NOT official RULER scores',
                construction='No test-specific model training; seed 20260927; balanced answers per cell',
                controls='128-token recall baseline; paired clean/distractor examples',
                length_policy='Execution budgets; canonical whole-text tokenization, actual prompt lengths and fact offsets retained',
                limitation='64 examples per cell; synthetic task/template scope; 4096 is extrapolation')
    (out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--vocab', default='vendor/nedo32k/surface-vocab.bin')
    p.add_argument('--output', default='lm100/benchmark_v2')
    p.add_argument('--per-cell', type=int, default=64)
    build(p.parse_args())
