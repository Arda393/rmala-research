"""Bounded GPU/runtime check; this is NOT the 3B-token training run."""
import json
import time
import torch

torch.set_num_threads(16)
print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(), stage="runtime")), flush=True)
from fla.layers.gated_deltanet import GatedDeltaNet
from fla.ops.gla import chunk_gla

torch.manual_seed(100)
layer = GatedDeltaNet(hidden_size=640, expand_v=1, head_dim=128, num_heads=5,
    use_gate=True, use_short_conv=True, conv_size=4, layer_idx=0,
    use_gdn_swa=True, gdn_swa_evict="betae", gdn_swa_window=64,
    gdn_swa_chunk=256, gdn_swa_gate_init=-4, gdn_swa_cache_norm="rms",
    gdn_swa_tau_init=1, gdn_swa_tau_freeze=True, gdn_swa_cache_kernel="sdpa").cuda()
x = torch.randn(2, 2048, 640, device="cuda", requires_grad=True)
for i in range(3):
    layer.zero_grad(set_to_none=True)
    x.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x)[0]
        loss = y.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    print(json.dumps(dict(stage="hola_layer", iteration=i, loss=loss.item(),
        seconds=time.perf_counter()-start, parameters=sum(p.numel() for p in layer.parameters()),
        max_memory_bytes=torch.cuda.max_memory_allocated())), flush=True)
q = torch.rand(2, 128, 10, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.rand_like(q, requires_grad=True)
v = torch.randn(2, 128, 10, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
g = torch.full_like(q, -.05, dtype=torch.float32, requires_grad=True)
y, _ = chunk_gla(q=q, k=k, v=v, g=g, scale=1., output_final_state=False)
y.float().square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (q,k,v,g))
print(json.dumps(dict(stage="gla_backward", status="PASS")), flush=True)
