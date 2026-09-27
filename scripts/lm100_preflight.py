"""Real-data ~100M technical check. Fresh main runs must restart at token zero."""
import gc
import hashlib
import json
import time
import unittest
from pathlib import Path
import numpy as np
import torch
from rmala.lm100_model import LM100, VARIANTS, normalized_gla
from rmala.lm100_data import TokenStream, sha_file
from rmala.lm100_compute import compute_estimate, collect_memory_stats
from rmala.attention import gla_reference


def digest_tensors(state):
    h=hashlib.sha256()
    for name,x in sorted(state.items()):
        if '.memory.' in name:
            continue
        h.update(name.encode())
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run():
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path('lm100/preflight')
    out.mkdir(parents=True,exist_ok=True)
    suite=unittest.TestSuite()
    for pattern in ('test_lm100_data.py','test_lm100_memory.py','test_lm100_train.py'):
        suite.addTests(unittest.defaultTestLoader.discover('tests',pattern=pattern))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Causal/data tests failed')
    print(json.dumps(dict(stage='unit_tests',status='PASS',tests=result.testsRun)),flush=True)
    # Verify appended-denominator FLA read and gradients against a serial recurrence.
    torch.manual_seed(122)
    inputs=[torch.rand(1,33,2,16,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
    inputs.append(torch.full((1,33,2,16),-.05,device='cuda',dtype=torch.float32,requires_grad=True))
    got,_=normalized_gla(*inputs)
    refs=[x.detach().clone().requires_grad_() for x in inputs]
    expected,_,_=gla_reference(*refs)
    torch.testing.assert_close(got,expected,atol=.012,rtol=.025)
    got.square().sum().backward()
    expected.square().sum().backward()
    for a,b in zip(inputs,refs):
        torch.testing.assert_close(a.grad.float(),b.grad.float(),atol=.03,rtol=.08)
    print(json.dumps(dict(stage='normalized_gla_output_gradient',status='PASS')),flush=True)
    gate_path='runs/v14_reference/post_gates.pt'
    gates=torch.load(gate_path,map_location='cpu',weights_only=False)
    gate=gates[str(('raw_int8',41001,1000))]
    lock=Path('lm100/data_3b_v2.lock.json')
    manifest=json.loads(lock.read_text())
    stream=TokenStream(manifest)
    # Fixed B=8, T=2048 for all variants, production batch uses accumulation=8.
    batch,length=8,2048
    raw=np.frombuffer(stream.read(0,batch*length+1),dtype='<u2').astype(np.int64)
    x=torch.from_numpy(raw[:-1].reshape(batch,length)).cuda()
    y=torch.from_numpy(raw[1:].reshape(batch,length)).cuda()
    rows=[]
    common_hash=None
    for variant in VARIANTS:
        print(json.dumps(dict(stage='model_begin',variant=variant)),flush=True)
        model=LM100(variant,gate_state=gate).cuda()
        count=sum(p.numel() for p in model.parameters())
        assert 98_000_000 <= count <= 103_000_000, count
        init_hash=digest_tensors(model.state_dict())
        if variant=='gla':
            common_hash=init_hash
        elif variant.startswith('v14'):
            assert init_hash==common_hash, 'GLA initialization mismatch'
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,fused=True)
        times=[]
        torch.cuda.reset_peak_memory_stats()
        for i in range(8):
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            begin=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model.losses(x,y).mean()
            assert torch.isfinite(loss), (variant,'nonfinite loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(grad), (variant,'nonfinite gradient')
            opt.step()
            torch.cuda.synchronize()
            seconds=time.perf_counter()-begin
            times.append(seconds)
            print(json.dumps(dict(stage='step',variant=variant,step=i,loss=float(loss),grad=float(grad),seconds=seconds)),flush=True)
        peak=torch.cuda.max_memory_allocated()
        ms=collect_memory_stats(model)
        if ms:
            assert all(r['reads']<=r['opportunities']*.05 and r['writes']<=r['opportunities']*.05 for r in ms)
        estimate=compute_estimate(model,batch,length,ms)
        # Profiler reports covered operators, not all custom-kernel FLOPs.
        opt.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True,record_shapes=True) as prof:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                ploss=model.losses(x,y).mean()
            ploss.backward()
        prof.export_chrome_trace(str(out/f'{variant}_trace.json'))
        covered=sum(e.flops for e in prof.key_averages())
        # Causality: full-model outputs before a suffix mutation must match.
        model.eval()
        # Cover several HoLA cache chunks, not only its first 256-token chunk.
        ids=x[:1].clone()
        changed=ids.clone()
        changed[:,1024:]=(changed[:,1024:]+37)%32000
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            before=model(ids)[:,:1024].float()
            after=model(changed)[:,:1024].float()
        torch.testing.assert_close(before,after,atol=.003,rtol=.003)
        row=dict(variant=variant,parameters=count,trunk_initial_sha256=init_hash,
            seconds_per_microbatch=sum(times[2:])/len(times[2:]),
            training_tokens_per_second=batch*length/(sum(times[2:])/len(times[2:])),
            max_gpu_allocated_bytes=peak,loss=float(loss),memory=ms,
            profiler_covered_flops=covered,compute=estimate,causality='PASS')
        rows.append(row)
        (out/f'{variant}.json').write_text(json.dumps(row,indent=2))
        del model,opt,prof,loss,ploss,before,after
        gc.collect()
        torch.cuda.empty_cache()
    source_files=['rmala/lm100_data.py','rmala/lm100_memory.py','rmala/lm100_model.py',
                  'rmala/lm100_compute.py','rmala/lm100_train.py','rmala/utility_gate.py','rmala/attention.py']
    report=dict(status='PASS',models=rows,batch_size=batch,sequence_length=length,
        gradient_accumulation=8,data_manifest_sha256=sha_file(lock),
        model_kwargs=dict(vocab_size=32000,dim=640,layers=16,heads=10,seed=41001),
        source_sha256={p:sha_file(p) for p in source_files},
        gate_checkpoint_sha256=sha_file(gate_path),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),main_training_started=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    run()
