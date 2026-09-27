"""Three real 100M updates with restart, exact data proof, and a tiny metric check."""
import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from rmala.lm100_data import TokenStream, canonical, sha_file
from rmala.lm100_model import LM100
from rmala.lm100_train import run,evaluate


def main():
    cfg=json.loads(Path('configs/lm100_3b.json').read_text())
    cfg.update(output_root='runs/lm100_runner_smoke_v2',checkpoint_every_updates=1,
               log_every_updates=1,profile_updates=[2])
    path=Path('lm100/runner_smoke_v2_config.json')
    if path.exists() and json.loads(path.read_text())!=cfg:
        raise RuntimeError('Different existing smoke configuration')
    path.write_text(canonical(cfg))
    args=SimpleNamespace(config=str(path),variant='gla',max_hours=1,stop_after_updates=2)
    run(args)
    gc.collect();torch.cuda.empty_cache()
    args.stop_after_updates=3
    run(args)
    gc.collect();torch.cuda.empty_cache()
    checkpoint=torch.load('runs/lm100_runner_smoke_v2/gla/latest.pt',map_location='cpu',weights_only=False)
    profile=json.loads(Path('runs/lm100_runner_smoke_v2/gla/profile_update_2_flops.json').read_text())
    assert profile['profiler_covered_flops']>0 and profile['cold_autotune_excluded']
    assert profile['cpu_process_peak_rss_bytes']<32*1024**3, 'Insufficient host RAM margin'
    assert checkpoint['state']['cursor']==3*cfg['tokens_per_update']
    manifest=json.loads(Path(cfg['data_manifest']).read_text())
    stream=TokenStream(manifest)
    chain='00'*32
    for i in range(3):
        h=hashlib.sha256(bytes.fromhex(chain))
        h.update(stream.read(i*cfg['tokens_per_update']+1,cfg['tokens_per_update']))
        chain=h.hexdigest()
    assert chain==checkpoint['state']['target_chain_sha256']
    from fla.modules import FusedCrossEntropyLoss
    torch.manual_seed(2)
    logits=torch.randn(5,32000,device='cuda',dtype=torch.bfloat16)
    targets=torch.tensor([4,15,-100,2,178],device='cuda')
    got=FusedCrossEntropyLoss(ignore_index=-100,reduction='none')(logits,targets)
    ref=F.cross_entropy(logits.float(),targets,ignore_index=-100,reduction='none')
    torch.testing.assert_close(got,ref,atol=2e-5,rtol=2e-5)
    model=LM100('gla').cuda()
    model.load_state_dict(checkpoint['model'])
    docs=[r for r in manifest['validation']['documents'] if 1<r['tokens']<=1024 and r['text_utf8_bytes']>0][:2]
    assert len(docs)==2
    manifest['validation']=dict(documents=docs,tokens_including_eos=sum(r['tokens'] for r in docs),
        content_tokens=sum(r['tokens']-1 for r in docs),text_utf8_bytes=sum(r['text_utf8_bytes'] for r in docs))
    metrics=evaluate(model,manifest,'validation',batch_size=2)
    assert all(math.isfinite(metrics[k]) and metrics[k]>0 for k in ('ppl','bpb','bpb_including_eos'))
    report=dict(status='PASS',updates=3,tokens=checkpoint['state']['cursor'],
        target_chain_sha256=chain,data_manifest_sha256=sha_file(cfg['data_manifest']),
        metrics_check=metrics,warm_profile=profile,main_training_started=False,
        source_sha256=checkpoint['state']['identity']['source_sha256'])
    Path('lm100/runner_smoke_v2_pass.json').write_text(canonical(report))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
