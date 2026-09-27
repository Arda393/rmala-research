# RMALA research

Residual-memory and gated linear-attention research: model implementations,
synthetic experiments, training, evaluation and tests. This repository contains
the research source, including the earlier experiment versions and the final
approximately 100M language-model comparison.

## Models

Five models were trained on exactly the same 3 billion target tokens: normalized
GLA, V14-LM full contribution, V14-LM half contribution, full attention and HoLA.
V14-LM did not establish a robust general advantage over GLA; the small half-
contribution improvement is from one training seed. Additional diagnostics are
not claimed as completed results here.

Weights, native GPU inference code, tokenizer and PPL/BPB/FLOP results:

| Model | Ethosoft | MercanAI |
|---|---|---|
| GLA | [model](https://huggingface.co/Ethosoft/rmala-gla-100m-3b) | [model](https://huggingface.co/MercanAI/rmala-gla-100m-3b) |
| V14 full | [model](https://huggingface.co/Ethosoft/rmala-v14-full-100m-3b) | [model](https://huggingface.co/MercanAI/rmala-v14-full-100m-3b) |
| V14 half | [model](https://huggingface.co/Ethosoft/rmala-v14-half-100m-3b) | [model](https://huggingface.co/MercanAI/rmala-v14-half-100m-3b) |
| Full attention | [model](https://huggingface.co/Ethosoft/rmala-full-attention-100m-3b) | [model](https://huggingface.co/MercanAI/rmala-full-attention-100m-3b) |
| HoLA | [model](https://huggingface.co/Ethosoft/rmala-hola-100m-3b) | [model](https://huggingface.co/MercanAI/rmala-hola-100m-3b) |

## Layout

- `rmala/`: attention, memory routing, gate, model, training and evaluation code.
- `scripts/`: experiments, diagnostics, checkpoint export and publication helpers.
- `configs/`: toy and research training configurations.
- `tests/`: unit and regression tests.
- [LM100_PROTOCOL.md](LM100_PROTOCOL.md): model and measurement definitions.

Execution logs, scheduler jobs/accounting, raw run reports, model weights,
credentials and dataset contents are not distributed in this Git repository.
Machine account names in example paths are replaced with `YOURUSER`.
Paths remain examples: configure your own data, dependencies and output locations.

## Small CPU example

```bash
python -m pip install -e .
python -m rmala.train --config configs/toy_gla.json --out runs/toy_gla --steps 10 --device cpu
python -m unittest discover -s tests
```

Some tests require optional dependencies or a GPU; individual standard-library
data/benchmark tests can run without torch.

## 100M GPU implementation

The tested stack is Python 3.11, torch 2.7.1 with CUDA 12.8, Triton 3.3.1,
transformers 4.57.6, einops 0.8.2 and numpy 2.2.6 on NVIDIA H100.
Use the pinned official HOLA/FLA source, not an arbitrary current FLA release:

```bash
git clone https://github.com/scale-lab-ai/HOLA vendor/HOLA
git -C vendor/HOLA checkout 7135e2291f18de17c10b59206641bd093efaac78
export PYTHONPATH="$PWD:$PWD/vendor/HOLA:$PYTHONPATH"
```

Training requires your own compatible pretokenized data, locked data manifest,
V14 initialization artifact and preflight report. These are required inputs,
not embedded datasets or automatically supplied checkpoints. Inspect
`scripts/lm100_preflight.py`, `configs/lm100_3b.json` and
`python -m rmala.lm100_data --help` before a run. Source hash guards intentionally
reject resuming a checkpoint with changed code; sanitizing machine paths does
not remove these guards.

For pretrained inference, download one of the linked model repositories and
follow its `inference.py` instructions. It uses native PyTorch/CUDA rather than
Transformers AutoModel registration. Its tokenizer vocabulary and native binary
are included in the model repository, with third-party notices.

## Diagnostic benchmark

`rmala/lm100_bench.py` builds original Turkish forced-choice tests for single-
and multi-key retrieval, distractors, position/context length, two-hop inference
and state tracking. `scripts/lm100_benchmark.py` evaluates common frozen models
and measures warm forward FLOPs/time. `scripts/lm100_benchmark_report.py`
collects results. These are not official RULER scores. The 4096-token condition
is extrapolation beyond the 2048-token training context. No test result is
implied merely by the presence of its code.

## Licensing

No new license grant for original project code or model weights is assigned by
this publication. External dependencies retain their respective licenses:
HOLA/FLA is MIT; NedoTokenizer is Apache-2.0. Their source is linked above or
distributed with the model repositories and its original notices.
