"""Reference inference for this native RMALA checkpoint (Linux NVIDIA CUDA)."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'vendor'/'HOLA'))
sys.path.insert(0, str(ROOT/'vendor'/'nedo32k'))


def load_model(model_dir=ROOT, device='cuda'):
    import torch
    from safetensors.torch import load_file
    from rmala.lm100_model import LM100
    root = Path(model_dir)
    cfg = json.loads((root/'config.json').read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    model = LM100(cfg['variant'], **cfg['model_kwargs'])
    state = load_file(str(root/'model.safetensors'), device='cpu')
    # These are tied parameters in the original architecture, saved once.
    state['head.weight'] = state['embedding.weight']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_tokenizer(model_dir=ROOT):
    import hashlib
    from nedotokenizer import SurfaceTokenizer
    root = Path(model_dir)
    raw = (root/'surface-vocab.bin').read_bytes()
    cfg = json.loads((root/'config.json').read_text())
    if hashlib.sha256(raw).hexdigest() != cfg['tokenizer_sha256']:
        raise ValueError('Tokenizer vocabulary checksum mismatch')
    return SurfaceTokenizer(raw)


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, default=ROOT)
    p.add_argument('--prompt', default='Bilim ve teknoloji')
    p.add_argument('--max-new-tokens', type=int, default=64)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This reference implementation requires an NVIDIA CUDA GPU.')
    tokenizer = load_tokenizer(args.model_dir)
    ids = [2] + list(tokenizer.encode_ids(args.prompt.encode('utf-8')))
    if not 0 < args.max_new_tokens <= 2048 or len(ids) >= 2048:
        raise ValueError('Use a prompt shorter than 2048 tokens and max-new-tokens in 1..2048.')
    model = load_model(args.model_dir)
    # Full-prefix recomputation preserves the original model and bank semantics.
    # This is a correctness-oriented reference, not an optimized cached decoder.
    with torch.no_grad():
        for _ in range(min(args.max_new_tokens, 2048-len(ids))):
            x = torch.tensor([ids], device='cuda', dtype=torch.long)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                token = int(model(x)[0,-1].argmax())
            if token == 2:
                break
            ids.append(token)
    raw = tokenizer.decode_ids(ids[1:])
    print(raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else raw)


if __name__ == '__main__':
    main()
