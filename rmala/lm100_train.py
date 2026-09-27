"""Resumable, common-stream 3B training and document-complete PPL/BPB."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import resource
import signal
import time
from pathlib import Path
import numpy as np
import torch
from .lm100_data import HEADER, TokenStream, canonical, sha_file
from .lm100_model import LM100
from .lm100_compute import compute_estimate, collect_memory_stats

SOURCE_FILES = ['rmala/lm100_data.py','rmala/lm100_memory.py','rmala/lm100_model.py',
                'rmala/lm100_compute.py','rmala/lm100_train.py','rmala/utility_gate.py','rmala/attention.py']


def vendor_revision():
    # The training container has no git executable. Read this pinned local clone's
    # HEAD/ref without a runtime package dependency; support loose and packed refs.
    gitdir=Path('vendor/HOLA/.git').resolve()
    head=(gitdir/'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref=head[5:]
        path=(gitdir/ref).resolve()
        if not path.is_relative_to(gitdir):
            raise RuntimeError('Invalid vendor Git reference')
        if path.exists():
            head=path.read_text().strip()
        else:
            refs={line.split()[1]:line.split()[0] for line in (gitdir/'packed-refs').read_text().splitlines()
                  if line and not line.startswith(('#','^'))}
            head=refs[ref]
    if len(head)!=40 or any(c not in '0123456789abcdef' for c in head):
        raise RuntimeError('Invalid vendor revision')
    return head


def atomic_json(path, value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(canonical(value),encoding='utf-8')
    os.replace(temp,path)


def flop_profiler():
    # FLOP estimates come from PyTorch operator shapes, not CUPTI kernel events.
    # Never capture cold Triton autotuning: its many benchmark launches can make
    # the profiler consume tens of GB of host RAM. GPU time is measured separately.
    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                  with_flops=True,record_shapes=True)


def learning_rate(cursor,cfg):
    if cursor < cfg['warmup_tokens']:
        return cfg['learning_rate']*max(1,cursor)/cfg['warmup_tokens']
    p=(cursor-cfg['warmup_tokens'])/(cfg['target_tokens']-cfg['warmup_tokens'])
    return cfg['minimum_learning_rate']+.5*(cfg['learning_rate']-cfg['minimum_learning_rate'])*(1+math.cos(math.pi*min(1,p)))


def training_batch(stream,cursor,count,length,device='cuda'):
    raw=stream.read(cursor,count+1)
    ids=np.frombuffer(raw,dtype='<u2').astype(np.int64)
    batch=math.ceil(count/length)
    x=np.full(batch*length,2,dtype=np.int64)
    y=np.full(batch*length,-100,dtype=np.int64)
    x[:count],y[:count]=ids[:-1],ids[1:]
    if ids.max()>=32000:
        raise ValueError('Token outside vocabulary')
    return (torch.from_numpy(x.reshape(batch,length)).to(device),
            torch.from_numpy(y.reshape(batch,length)).to(device),raw[2:])


def evaluation_chunks(manifest,split,length,minimum_tokens=None):
    total=0
    for record in manifest[split]['documents']:
        path=Path(manifest['dataset_root'])/'shards'/record['shard']
        with path.open('rb') as f:
            f.seek(HEADER+2*record['offset'])
            raw=f.read(2*record['tokens'])
        if hashlib.sha256(raw).hexdigest()!=record['token_sha256']:
            raise ValueError(f'Evaluation document changed: {path}')
        ids=np.frombuffer(raw,dtype='<u2').astype(np.int64)
        for start in range(0,len(ids),length):
            end=min(start+length,len(ids))
            targets=ids[start:end]
            inputs=np.concatenate(([manifest['eval_start_token'] if start==0 else ids[start-1]],targets[:-1]))
            content=np.ones(end-start,dtype=bool)
            if end==len(ids):
                content[-1]=False  # only the terminal document EOS is excluded
            yield inputs,targets,content,record['text_utf8_bytes'] if end==len(ids) else 0
        total+=len(ids)
        if minimum_tokens is not None and total>=minimum_tokens:
            break  # never truncate a document or infer its byte count


@torch.no_grad()
def evaluate(model,manifest,split,batch_size=8,length=2048,minimum_tokens=None):
    was_training=model.training
    model.eval()
    total_nll=content_nll=0.
    tokens=content_tokens=nbytes=batches=0
    inference_flops=0
    inference_dense=0
    profiler_covered_flops=None
    queue=[]
    start_time=time.perf_counter()

    def process(items):
        nonlocal inference_flops,inference_dense,profiler_covered_flops
        x=np.full((len(items),length),2,dtype=np.int64)
        y=np.full_like(x,-100)
        mask=np.zeros_like(x,dtype=bool)
        for i,(inputs,targets,content,_) in enumerate(items):
            n=len(targets)
            x[i,:n],y[i,:n],mask[i,:n]=inputs,targets,content
        targets=torch.from_numpy(y).cuda()
        prof=None
        if batches==2 and minimum_tokens is None and len(items)==batch_size:
            prof=flop_profiler()
            prof.__enter__()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            losses=model.losses(torch.from_numpy(x).cuda(),targets)
        if prof is not None:
            prof.__exit__(None,None,None)
            profiler_covered_flops=sum(e.flops for e in prof.key_averages())
        estimate=compute_estimate(model,len(items),length,collect_memory_stats(model),training=False)
        inference_flops+=estimate['total_algorithmic_flops_estimate']
        inference_dense+=estimate['dense_matmul_flops']
        return (float(losses.double().sum()),float(losses[torch.from_numpy(mask).cuda()].double().sum()),
                int((y!=-100).sum()),int(mask.sum()),sum(item[3] for item in items))

    for chunk in evaluation_chunks(manifest,split,length,minimum_tokens):
        queue.append(chunk)
        if len(queue)==batch_size:
            a,b,c,d,e=process(queue)
            total_nll+=a;content_nll+=b;tokens+=c;content_tokens+=d;nbytes+=e;batches+=1
            queue=[]
    if queue:
        a,b,c,d,e=process(queue)
        total_nll+=a;content_nll+=b;tokens+=c;content_tokens+=d;nbytes+=e;batches+=1
    if minimum_tokens is None:
        assert tokens==manifest[split]['tokens_including_eos']
        assert content_tokens==manifest[split]['content_tokens']
        assert nbytes==manifest[split]['text_utf8_bytes']
    model.train(was_training)
    return dict(split=split,tokens=tokens,content_tokens=content_tokens,text_utf8_bytes=nbytes,
        total_nll=total_nll,content_nll=content_nll,ppl=math.exp(total_nll/tokens),
        bpb=content_nll/(math.log(2)*nbytes),bpb_including_eos=total_nll/(math.log(2)*nbytes),
        seconds=time.perf_counter()-start_time,
        batches=batches,subset=minimum_tokens is not None,
        inference_algorithmic_flops_estimate=inference_flops,inference_dense_matmul_flops=inference_dense,
        sample_profiler_covered_flops=profiler_covered_flops,
        profiled_batch_index=2 if profiler_covered_flops is not None else None,
        compute_scope='Includes padded positions actually executed; logical attention estimate, not complete hardware FLOPs')


def save_checkpoint(out,model,opt,state):
    temp=out/'checkpoint.tmp'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
    current=out/'latest.pt'
    if current.exists():
        os.replace(current,out/'previous.pt')
    os.replace(temp,current)
    atomic_json(out/'progress.json',state)


def run(args):
    cfg=json.loads(Path(args.config).read_text())
    assert args.variant in cfg['variants']
    manifest_path=Path(cfg['data_manifest'])
    manifest=json.loads(manifest_path.read_text())
    assert cfg['target_tokens']==manifest['train_target_tokens']==3_000_000_000
    preflight=json.loads(Path(cfg['preflight_report']).read_text())
    assert preflight['status']=='PASS', 'Technical preflight required'
    assert all(cfg[k]==v for k,v in preflight['model_kwargs'].items())
    assert preflight['data_manifest_sha256']==sha_file(manifest_path)
    assert preflight['batch_size']==cfg['microbatch_size'] and preflight['sequence_length']==cfg['sequence_length']
    gate_path=Path(cfg['gate_checkpoint'])
    assert preflight['gate_checkpoint_sha256']==sha_file(gate_path)
    for path,expected in preflight['source_sha256'].items():
        if sha_file(path)!=expected:
            raise RuntimeError(f'Technically verified source changed: {path}')
    vendor_commit=vendor_revision()
    if vendor_commit!='7135e2291f18de17c10b59206641bd093efaac78':
        raise RuntimeError('HoLA vendor revision changed')
    out=Path(cfg['output_root'])/args.variant
    out.mkdir(parents=True,exist_ok=True)
    # OS lock is released on process exit, including crashes, without a stale PID file.
    lock=(out/'process.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=dict(config_sha256=sha_file(args.config),data_sha256=sha_file(manifest_path),
        gate_sha256=sha_file(gate_path),source_sha256={p:sha_file(p) for p in SOURCE_FILES},
        variant=args.variant,hola_commit='7135e2291f18de17c10b59206641bd093efaac78')
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32=False  # calibrated FP32 gate; dense trunk uses BF16
    gate=torch.load(gate_path,map_location='cpu',weights_only=False)[cfg['gate_key']]
    model=LM100(args.variant,vocab_size=cfg['vocab_size'],dim=cfg['dim'],layers=cfg['layers'],
        heads=cfg['heads'],seed=cfg['seed'],gate_state=gate).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),
                         weight_decay=cfg['weight_decay'],fused=True)
    state=dict(identity=identity,cursor=0,updates=0,target_chain_sha256='00'*32,
        cumulative_algorithmic_flops_estimate=0,cumulative_dense_matmul_flops=0,
        train_seconds=0.,status='training',last_validation_cursor=0,
        parameters=sum(p.numel() for p in model.parameters()))
    if (out/'latest.pt').exists():
        saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if saved['state']['identity']!=identity:
            raise RuntimeError('Resume identity changed; refusing mixed experiment')
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        state=saved['state']
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    else:
        save_checkpoint(out,model,opt,state)
    if state['status']=='completed':
        print(canonical(state),flush=True)
        return
    state['status']='training'
    stream=TokenStream(manifest)
    print(json.dumps(dict(stage='verify_locked_data',variant=args.variant,cursor=state['cursor'])),flush=True)
    # Recheck the locked raw stream once per job. This catches data changes even
    # if file names and sizes remain unchanged, before any new optimizer update.
    if stream.digest()!=manifest['train_input_stream_sha256']:
        raise RuntimeError('Locked training token bytes changed')
    print(json.dumps(dict(stage='training_started',variant=args.variant,cursor=state['cursor'],
                         target_tokens=cfg['target_tokens'])),flush=True)
    interrupted=[False]
    signal.signal(signal.SIGUSR1,lambda *unused: interrupted.__setitem__(0,True))
    signal.signal(signal.SIGTERM,lambda *unused: interrupted.__setitem__(0,True))
    job_start=time.perf_counter()
    completed_in_process=0
    length=cfg['sequence_length']
    micro=cfg['microbatch_size']*length
    window=dict(tokens=0,seconds=0.,loss_sum=0.,writes=0,reads=0,accepted=0,opportunities=0)
    with (out/'training.jsonl').open('a',buffering=1) as log:
        while state['cursor']<cfg['target_tokens']:
            step_tokens=min(cfg['tokens_per_update'],cfg['target_tokens']-state['cursor'])
            lr=learning_rate(state['cursor']+step_tokens,cfg)
            for group in opt.param_groups:
                group['lr']=lr
            opt.zero_grad(set_to_none=True)
            start_cursor=state['cursor']
            chain=hashlib.sha256(bytes.fromhex(state['target_chain_sha256']))
            loss_sum=0.
            flops=dense=0
            torch.cuda.synchronize()
            begin=time.perf_counter()
            prof=None
            if state['updates']+1 in cfg['profile_updates'] and completed_in_process>=1:
                prof=flop_profiler()
            for offset in range(0,step_tokens,micro):
                n=min(micro,step_tokens-offset)
                x,y,target_bytes=training_batch(stream,start_cursor+offset,n,length)
                chain.update(target_bytes)
                if prof is not None and offset==0:
                    prof.__enter__()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    losses=model.losses(x,y)
                    loss=losses.sum()/step_tokens
                loss.backward()
                if prof is not None and offset==0:
                    prof.__exit__(None,None,None)
                loss_sum+=float(loss.detach())*step_tokens
                stats=collect_memory_stats(model)
                estimate=compute_estimate(model,x.shape[0],length,stats)
                flops+=estimate['total_algorithmic_flops_estimate'];dense+=estimate['dense_matmul_flops']
                for row in stats:
                    for key in ('writes','reads','accepted','opportunities'):
                        window[key]+=row[key]
                del losses,loss,x,y
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            if not torch.isfinite(grad) or not math.isfinite(loss_sum):
                atomic_json(out/'failure.json',dict(cursor=start_cursor,reason='nonfinite loss/gradient'))
                raise RuntimeError('Nonfinite training update; latest checkpoint preserved')
            opt.step()
            torch.cuda.synchronize()
            seconds=time.perf_counter()-begin
            state['cursor']+=step_tokens;state['updates']+=1
            completed_in_process+=1
            state['target_chain_sha256']=chain.hexdigest()
            state['cumulative_algorithmic_flops_estimate']+=flops
            state['cumulative_dense_matmul_flops']+=dense
            state['train_seconds']+=seconds
            window['tokens']+=step_tokens;window['seconds']+=seconds;window['loss_sum']+=loss_sum
            if prof is not None:
                atomic_json(out/f'profile_update_{state["updates"]}_flops.json',dict(
                    profiler_covered_flops=sum(e.flops for e in prof.key_averages()),
                    scope='warm first microbatch forward+backward; PyTorch operator shapes only; Triton kernels uncovered',
                    cold_autotune_excluded=True,chrome_trace_exported=False,
                    cpu_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    microbatch_tokens=micro,update=state['updates']))
                del prof
            if state['updates']==1 or state['updates']%cfg['log_every_updates']==0 or state['cursor']==cfg['target_tokens']:
                row=dict(update=state['updates'],tokens=state['cursor'],loss=window['loss_sum']/window['tokens'],
                    lr=lr,grad_norm=float(grad),tokens_per_second=window['tokens']/window['seconds'],
                    max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cpu_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    cumulative_algorithmic_flops_estimate=state['cumulative_algorithmic_flops_estimate'],
                    cumulative_dense_matmul_flops=state['cumulative_dense_matmul_flops'],
                    target_chain_sha256=state['target_chain_sha256'],memory={k:window[k] for k in ('writes','reads','accepted','opportunities')})
                log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                atomic_json(out/'progress.json',dict(**state,last_metrics=row))
                window=dict(tokens=0,seconds=0.,loss_sum=0.,writes=0,reads=0,accepted=0,opportunities=0)
            due_eval=state['cursor']-state['last_validation_cursor']>=cfg['validation_every_tokens']
            if due_eval and state['cursor']<cfg['target_tokens']:
                result=evaluate(model,manifest,'validation',cfg['microbatch_size'],length,cfg['quick_validation_tokens'])
                result['trained_tokens']=state['cursor']
                atomic_json(out/f'validation_{state["cursor"]}.json',result)
                state['last_validation_cursor']=state['cursor']
            stop=interrupted[0] or time.perf_counter()-job_start>=args.max_hours*3600
            if args.stop_after_updates and state['updates']>=args.stop_after_updates:
                stop=True
            if state['updates']%cfg['checkpoint_every_updates']==0 or stop or state['cursor']==cfg['target_tokens'] or due_eval:
                state['status']='needs_resume' if stop else 'training'
                save_checkpoint(out,model,opt,state)
            if stop:
                print(canonical(state),flush=True)
                return
    results={split:evaluate(model,manifest,split,cfg['microbatch_size'],length) for split in ('validation','test')}
    atomic_json(out/'evaluation.json',dict(identity=identity,trained_tokens=state['cursor'],
        target_chain_sha256=state['target_chain_sha256'],results=results,
        cumulative_algorithmic_flops_estimate=state['cumulative_algorithmic_flops_estimate'],
        cumulative_dense_matmul_flops=state['cumulative_dense_matmul_flops']))
    state['status']='completed'
    save_checkpoint(out,model,opt,state)
    print(canonical(dict(state=state,results=results)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/lm100_3b.json')
    parser.add_argument('--variant',required=True)
    parser.add_argument('--max-hours',type=float,default=68.)
    parser.add_argument('--stop-after-updates',type=int,default=0)
    run(parser.parse_args())
