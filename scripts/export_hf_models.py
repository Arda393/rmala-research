"""Export only final research model weights and a reviewed inference allowlist."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import torch
from safetensors.torch import save_file, load_file

ROOT=Path.cwd()
OUTPUT=ROOT/'exports'/'hf_lm100_3b'
NAMES={'gla':'rmala-gla-100m-3b','v14_full':'rmala-v14-full-100m-3b',
       'v14_half':'rmala-v14-half-100m-3b','full':'rmala-full-attention-100m-3b',
       'hola':'rmala-hola-100m-3b'}
SOURCES=['__init__.py','lm100_model.py','lm100_memory.py','attention.py','bank.py',
         'utility_gate.py','rejection.py','budget_memory.py']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    torch.set_num_threads(2)
    cfg=json.loads((ROOT/'configs/lm100_3b.json').read_text())
    evaluations={v:json.loads((ROOT/cfg['output_root']/v/'evaluation.json').read_text()) for v in NAMES}
    compare='\n'.join(f"| {v} | {e['results']['test']['ppl']:.6f} | {e['results']['test']['bpb']:.6f} | {e['cumulative_algorithmic_flops_estimate']:.6e} |" for v,e in evaluations.items())
    vendor=ROOT/'vendor/HOLA'
    vendor_files=json.loads((ROOT/'lm100/hola_export_files.json').read_text())
    export_reports=[]
    for variant,name in NAMES.items():
        out=OUTPUT/name
        out.mkdir(parents=True,exist_ok=True)
        path=ROOT/cfg['output_root']/variant/'latest.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        state=saved['state']; weights=saved['model']
        assert state['status']=='completed' and state['cursor']==3000000000
        assert state['identity']==evaluations[variant]['identity']
        assert state['identity']['config_sha256']==sha(ROOT/'configs/lm100_3b.json')
        for source,expected in state['identity']['source_sha256'].items():
            assert sha(ROOT/source)==expected,source
        assert torch.equal(weights['head.weight'],weights['embedding.weight'])
        export_weights={k:v.contiguous() for k,v in weights.items() if k!='head.weight'}
        save_file(export_weights,str(out/'model.safetensors'),metadata={'format':'pt','head.weight':'embedding.weight'})
        reloaded=load_file(str(out/'model.safetensors'))
        assert reloaded.keys()==export_weights.keys()
        for key,value in reloaded.items():
            assert torch.equal(value,weights[key]),key
        del reloaded,export_weights
        for source in SOURCES:
            dest=out/'rmala'/source;dest.parent.mkdir(exist_ok=True,parents=True)
            shutil.copyfile(ROOT/'rmala'/source,dest)
        for source in vendor_files+['LICENSE']:
            src=vendor/source
            if src.is_file() and not src.is_symlink():
                dest=out/'vendor/HOLA'/source;dest.parent.mkdir(exist_ok=True,parents=True)
                shutil.copyfile(src,dest)
        for source in ['__init__.py','_native.abi3.so']:
            dest=out/'vendor/nedo32k/nedotokenizer'/source;dest.parent.mkdir(exist_ok=True,parents=True)
            shutil.copyfile(ROOT/'vendor/nedo32k/nedotokenizer'/source,dest)
        shutil.copyfile(ROOT/'vendor/nedo32k/surface-vocab.bin',out/'surface-vocab.bin')
        for source in ['LICENSE','THIRD_PARTY_LICENSES/ZEMBEREK_NOTICE.md']:
            dest=out/'vendor/nedo32k'/source;dest.parent.mkdir(exist_ok=True,parents=True)
            shutil.copyfile(Path('/arf/scratch/YOURUSER/NedoTokenizer')/source,dest)
        shutil.copyfile(ROOT/'scripts/hf_inference_template.py',out/'inference.py')
        for doc in ['LM100_PROTOCOL.md','LM100_RESULTS.md']:
            shutil.copyfile(ROOT/doc,out/doc)
        model_cfg=dict(model_type='rmala_research',architectures=['LM100'],variant=variant,
            vocab_size=32000,torch_dtype='float32',max_position_embeddings=2048,
            model_kwargs={k:cfg[k] for k in ['vocab_size','dim','layers','heads','seed']},
            eos_token_id=2,bos_token_id=1,pad_token_id=0,
            inference_start_token_id=2,trained_tokens=state['cursor'],
            tokenizer_sha256=sha(out/'surface-vocab.bin'),
            hola_commit=state['identity']['hola_commit'],
            native_loader='inference.load_model; not registered with Transformers AutoModel')
        write_json(out/'config.json',model_cfg)
        # Explicit training settings without cluster paths, private corpora, or optimizer/RNG state.
        training={k:v for k,v in cfg.items() if k not in ['data_manifest','preflight_report','gate_checkpoint','output_root']}
        write_json(out/'training_config.json',training)
        evaluation=evaluations[variant]
        write_json(out/'evaluation.json',evaluation)
        write_json(out/'training_summary.json',dict(variant=variant,parameters=state['parameters'],
            trained_tokens=state['cursor'],updates=state['updates'],train_seconds=state['train_seconds'],
            target_chain_sha256=state['target_chain_sha256'],identity=state['identity']))
        (out/'requirements.txt').write_text('torch==2.7.1\ntriton==3.3.1\ntransformers==4.57.6\neinops==0.8.2\nnumpy==2.2.6\nsafetensors>=0.5\nhuggingface_hub>=0.36\n')
        (out/'THIRD_PARTY_NOTICES.md').write_text(
            '# Third-party components\n\n'
            'HOLA / Flash Linear Attention source is bundled at vendor/HOLA, pinned to '
            '7135e2291f18de17c10b59206641bd093efaac78 from https://github.com/scale-lab-ai/HOLA. '
            'Its MIT license and source copyright notices are retained.\n\n'
            'NedoTokenizer Python wrapper and native Linux x86_64 CPython 3.11+ ABI3 binary '
            'are bundled at vendor/nedo32k, with its Apache-2.0 notice and Zemberek notice. '
            'Source: https://github.com/ethosoftai/NedoTokenizer. The vocabulary is verified by SHA-256 '
            'in config.json. Binary and vocabulary checksums are in SHA256SUMS.json.\n\n'
            'No new license grant for the model weights or original RMALA project code is specified '
            'in this release. Third-party licenses apply to their respective components.\n',encoding='utf-8')
        card=f'''---
language:
- tr
tags:
- rmala
- research
- causal-language-model
- linear-attention
model-index:
- name: {name}
  results: []
---
# {name}

Final **{state['parameters']:,}-parameter** Turkish base language-model research checkpoint,
trained from scratch on exactly **3,000,000,000 target tokens**. Variant: **{variant}**.
This is a base model, not an instruction-tuned or chat model.

Mirrors: [Ethosoft/{name}](https://huggingface.co/Ethosoft/{name}) and
[MercanAI/{name}](https://huggingface.co/MercanAI/{name}).

## Results and scope

All five variants used the same tokenizer, target tokens in the same order, and held-out
validation/test sets. One training seed (41001) was used. Lower PPL/BPB is better.

| Variant | Test PPL | Test BPB | Total training algorithmic FLOP estimate |
|---|---:|---:|---:|
{compare}

PPL includes terminal EOS; BPB uses content NLL divided by original UTF-8 byte count
and excludes terminal EOS. Complete-document evaluation uses 2048-token chunks.
Test: 11,352,596 tokens / 39,843,759 original UTF-8 bytes. Full validation/test metrics
are in evaluation.json. Training and inference FLOP estimates are **not full hardware
FLOP measurements**: excluded operations and profiler coverage are documented in LM100_PROTOCOL.md.

The V14-half gain over GLA is small and does not establish robust superiority. V14-full
did not improve test PPL. No general harmlessness or production-readiness claim is made.
Long-range retrieval/reasoning diagnostics are not included as completed results in this release.
HoLA uses the official pinned GatedDeltaNet/cache implementation; it is a different backbone
from normalized GLA, so this is not a cache-only ablation.

## Use (native PyTorch, Linux NVIDIA CUDA)

This architecture is **not registered with Transformers AutoModel**. Use the bundled loader.
FP32 checkpoint values are preserved exactly; the tested compute mode uses BF16 autocast
and disables TF32. The bundled tokenizer native binary targets Linux x86_64 / CPython 3.11+.

```bash
hf download MercanAI/{name} --local-dir {name}
cd {name}
pip install -r requirements.txt
python inference.py --prompt "Bilim ve teknoloji" --max-new-tokens 64
```

The reference decoder recomputes the full prefix and stops at the 2048-token context
limit; it is not an optimized KV-cache decoder. First execution compiles Triton kernels.
The original training model sources and pinned HOLA/FLA sources are bundled.

## Training and limitations

16 layers, width 640, tied 32K input/output embeddings, RMSNorm and SwiGLU.
GLA/V14: 10 heads of size 64. Full attention: causal SDPA and RoPE. HoLA: 5 heads
of size 128, official betae cache, window 64 and chunk size 256. Context: 2048.
Dataset: a fixed 3B-token subset of the MercanSet V11 / MercanPretraining pretokenized
collection, with shard-disjoint validation/test. Cross-collection text deduplication was
not performed; absolute contamination-freedom is not claimed. Dataset files are not redistributed.
See training_config.json and LM100_PROTOCOL.md for optimization and V14-LM adaptation details.

V14-LM is an adaptation of the earlier synthetic V14 gate: contextual keys, int8 values,
per-head 2048-byte bank budget and 5% read/write admission limits. A learned straight-through
gate uses a hard 0.99 forward threshold, with accepted memory alpha=1 or 0.5. This does not
imply a 95% reduction in whole-model FLOPs or that accepted reads are correct.

Only final model tensors are distributed; optimizer states, credentials and training text
are not included. The tied head is stored once and restored by inference.py.
Weights/code licensing is not newly assigned by this upload; see THIRD_PARTY_NOTICES.md.
'''
        (out/'README.md').write_text(card,encoding='utf-8')
        checksums={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
        write_json(out/'SHA256SUMS.json',checksums)
        report=dict(variant=variant,name=name,path=str(out),tensor_keys=len(weights),
            saved_tensor_keys=len(weights)-1,parameters=state['parameters'],
            checkpoint_sha256=sha(path),safetensors_sha256=sha(out/'model.safetensors'),
            bytes=(out/'model.safetensors').stat().st_size,weight_roundtrip_exact=True,
            files=len(checksums)+1)
        export_reports.append(report)
        print(json.dumps(report),flush=True)
        del saved,weights
        gc.collect()
    write_json(OUTPUT/'export_report.json',export_reports)


if __name__=='__main__':main()
