"""Immutable common token stream and document-complete MercanSet evaluation.

Binary format: 96-byte headers; tokens <u2; document records <QQIIII.
No tokenization, re-sampling or model-specific data selection is performed.
This module intentionally uses only Python's standard library.
"""
import argparse
import bisect
import hashlib
import json
import random
import struct
from pathlib import Path

HEADER = 96
INDEX = struct.Struct('<QQIIII')
DEFAULT_ROOT = '/arf/scratch/YOURUSER/mercanset_v11_nedo32k_pretokenized_20260902'


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def check_header(path, magic, expected_size):
    p = Path(path)
    if p.stat().st_size != expected_size:
        raise ValueError(f'Unexpected size: {p}')
    with p.open('rb') as f:
        raw = f.read(HEADER)
    if len(raw) != HEADER or struct.unpack('<8sII', raw[:16]) != (magic, 1, HEADER):
        raise ValueError(f'Invalid binary header: {p}')


def shard_record(root, path, vocab_hash):
    m = json.loads(path.read_text(encoding='utf-8'))
    if m['status'] != 'PASS' or m['vocab_sha256'] != vocab_hash:
        raise ValueError(f'Invalid shard manifest: {path}')
    # File basenames only; manifests cannot escape their dataset directory.
    for key in ('tokens_file', 'index_file'):
        if Path(m[key]).name != m[key]:
            raise ValueError('Expected a shard basename')
    n, d = int(m['tokens_including_eos']), int(m['documents'])
    check_header(root/'shards'/m['tokens_file'], b'NDTKTOK1', HEADER + 2*n)
    check_header(root/'shards'/m['index_file'], b'NDTKIDX1', HEADER + INDEX.size*d)
    return dict(manifest=path.name, manifest_sha256=sha_file(path),
        tokens_file=m['tokens_file'], index_file=m['index_file'],
        tokens=n, documents=d, text_utf8_bytes=m['text_utf8_bytes'],
        tokens_sha256=m['tokens_sha256'], index_sha256=m['index_sha256'])


def select_eval(root, records, target_tokens, seed):
    rng = random.Random(seed)
    result, total_tokens, total_bytes = [], 0, 0
    quota = (target_tokens + len(records)-1)//len(records)
    for shard in records:
        ip = root/'shards'/shard['index_file']
        if sha_file(ip) != shard['index_sha256']:
            raise ValueError(f'Index checksum mismatch: {ip}')
        raw = ip.read_bytes()[HEADER:]
        entries = list(INDEX.iter_unpack(raw))
        if sum(x[2] for x in entries) != shard['tokens'] or sum(x[3] for x in entries) != shard['text_utf8_bytes']:
            raise ValueError('Index totals do not match manifest')
        expected_offset = 0
        for _, offset, count, _, _, _ in entries:
            if count < 1 or offset != expected_offset:
                raise ValueError('Non-contiguous or empty document record')
            expected_offset += count
        order = list(range(len(entries)))
        rng.shuffle(order)
        subtotal = 0
        with (root/'shards'/shard['tokens_file']).open('rb') as f:
            for idx in order:
                doc_id, offset, count, nbytes, _, _ = entries[idx]
                f.seek(HEADER + 2*offset)
                payload = f.read(2*count)
                if len(payload) != 2*count or payload[-2:] != b'\x02\x00':
                    raise ValueError('Truncated document or missing EOS')
                result.append(dict(shard=shard['tokens_file'], index_row=idx,
                    document_id=doc_id, offset=offset, tokens=count,
                    text_utf8_bytes=nbytes, token_sha256=hashlib.sha256(payload).hexdigest()))
                subtotal += count
                total_tokens += count
                total_bytes += nbytes
                if subtotal >= quota:
                    break
    if total_tokens < target_tokens:
        raise ValueError('Insufficient complete documents for evaluation budget')
    return dict(documents=result, tokens_including_eos=total_tokens,
        content_tokens=total_tokens-len(result), text_utf8_bytes=total_bytes)


class TokenStream:
    def __init__(self, manifest):
        self.manifest = manifest
        self.root = Path(manifest['dataset_root'])/'shards'
        self.segments = manifest['train_segments']
        self.ends, total = [], 0
        for segment in self.segments:
            total += segment['take_tokens']
            self.ends.append(total)
        self.total = total

    def read(self, start, count):
        if start < 0 or count < 0 or start+count > self.total:
            raise ValueError('Read outside locked training stream')
        chunks = []
        while count:
            i = bisect.bisect_right(self.ends, start)
            begin = self.ends[i-1] if i else 0
            n = min(count, self.ends[i]-start)
            segment = self.segments[i]
            with (self.root/segment['tokens_file']).open('rb') as f:
                f.seek(HEADER + 2*(segment.get('offset', 0)+start-begin))
                payload = f.read(2*n)
            if len(payload) != 2*n:
                raise ValueError('Training stream was truncated')
            chunks.append(payload)
            start, count = start+n, count-n
        return b''.join(chunks)

    def digest(self):
        h = hashlib.sha256()
        for offset in range(0, self.total, 4 << 20):
            h.update(self.read(offset, min(4 << 20, self.total-offset)))
        return h.hexdigest()


def build_manifest(root, train_tokens=3_000_000_000, eval_tokens=10_000_000,
                   holdout_shards=32, seed=20260926):
    root = Path(root).resolve()
    parent = json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    if parent['status'] != 'PASS' or parent['vocab_size'] != 32000:
        raise ValueError('Unexpected dataset manifest')
    paths = sorted((root/'shards').glob('*.manifest.json'))
    # Two sealed collections share this directory, with overlapping part numbers.
    # Never identify a shard by its numeric prefix or silently ignore extras.
    parents = {'nedotokenizer_pretokenized_shard_v1': ('MANIFEST.json', parent)}
    extra = root/'MercanPretraining_MANIFEST.json'
    if extra.exists():
        parents['nedotokenizer_pretokenized_shard_v2'] = (extra.name, json.loads(extra.read_text(encoding='utf-8')))
    totals = {key: dict(shards=0, tokens_including_eos=0, documents=0, text_utf8_bytes=0) for key in parents}
    for path in paths:
        sm = json.loads(path.read_text(encoding='utf-8'))
        schema = sm['schema']
        if schema not in totals or sm['status'] != 'PASS' or sm['vocab_sha256'] != parent['vocab_sha256']:
            raise ValueError(f'Unknown or incompatible shard: {path}')
        totals[schema]['shards'] += 1
        for key in ('tokens_including_eos', 'documents', 'text_utf8_bytes'):
            totals[schema][key] += sm[key]
    for schema, (_, pm) in parents.items():
        if pm['status'] != 'PASS' or any(totals[schema][k] != pm[k] for k in totals[schema]):
            raise ValueError(f'Sealed collection totals mismatch: {schema}')
    if len(paths) <= 2*holdout_shards:
        raise ValueError('Too large holdout')
    random.Random(seed).shuffle(paths)
    val_paths = paths[:holdout_shards]
    test_paths = paths[holdout_shards:2*holdout_shards]
    train_paths = paths[2*holdout_shards:]
    records, segments, need = [], [], train_tokens+1
    for path in train_paths:
        r = shard_record(root, path, parent['vocab_sha256'])
        records.append(r)
        take = min(need, r['tokens'])
        segments.append(dict(tokens_file=r['tokens_file'], offset=0, take_tokens=take))
        need -= take
        if not need:
            break
    if need:
        raise ValueError('Not enough training data')
    val = [shard_record(root, p, parent['vocab_sha256']) for p in val_paths]
    test = [shard_record(root, p, parent['vocab_sha256']) for p in test_paths]
    manifest = dict(schema='rmala_lm100_common_data_v2', dataset_root=str(root),
        dataset_manifest_sha256=sha_file(root/'MANIFEST.json'),
        collections={schema: dict(manifest=name, manifest_sha256=sha_file(root/name), **totals[schema])
                     for schema, (name, _) in parents.items()},
        vocab_sha256=parent['vocab_sha256'], vocab_size=32000, bos=1, eos=2, eval_start_token=2,
        seed=seed, train_target_tokens=train_tokens,
        train_segments=segments, train_shards=records,
        validation_shards=val, test_shards=test,
        validation=select_eval(root, val, eval_tokens, seed+1),
        test=select_eval(root, test, eval_tokens, seed+2),
        train_policy='fixed shuffled shard order; packed EOS documents; block context resets; targets at stream offsets 1..N',
        eval_policy='complete documents; preceding EOS as start context (BOS not present in training); context resets every 2048 targets with previous token overlap; EOS in PPL, excluded from content BPB',
        contamination_scope='disjoint physical shards/documents; cross-shard text duplicates have not been deduplicated')
    manifest['train_input_stream_sha256'] = TokenStream(manifest).digest()
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=DEFAULT_ROOT)
    p.add_argument('--out', required=True)
    p.add_argument('--train-tokens', type=int, default=3_000_000_000)
    p.add_argument('--eval-tokens', type=int, default=10_000_000)
    p.add_argument('--holdout-shards', type=int, default=32)
    args = p.parse_args()
    result = build_manifest(args.root, args.train_tokens, args.eval_tokens, args.holdout_shards)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(result)
    if out.exists():
        if out.read_text(encoding='utf-8') != payload:
            raise RuntimeError('Refusing to replace an existing different data lock')
    else:
        with out.open('x', encoding='utf-8', newline='\n') as f:
            f.write(payload)
    print(canonical(dict(manifest=str(out), manifest_sha256=sha_file(out),
        train_target_tokens=result['train_target_tokens'],
        train_input_stream_sha256=result['train_input_stream_sha256'],
        validation_tokens=result['validation']['tokens_including_eos'],
        test_tokens=result['test']['tokens_including_eos'])), flush=True)


if __name__ == '__main__':
    main()
