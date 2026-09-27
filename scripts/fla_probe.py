"""Run under the pinned FLA/container environment on an allocated GPU."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rmala.attention import gla_fla,gla_reference

torch.manual_seed(29)
records=[]
for dtype,tolerance in ((torch.float32,.004),(torch.bfloat16,.04)):
    inputs=[torch.rand(1,64,2,16,device="cuda",dtype=dtype).requires_grad_() for _ in range(2)]
    inputs.append(torch.randn(1,64,2,16,device="cuda",dtype=dtype).requires_grad_())
    inputs.append((-torch.rand(1,64,2,16,device="cuda")/16).requires_grad_())
    ref_inputs=[x.detach().clone().requires_grad_() for x in inputs]
    expected=gla_reference(*ref_inputs);actual=gla_fla(*inputs)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=tolerance,rtol=tolerance)
    sum(o.square().mean() for o in actual[:2]).backward()
    sum(o.square().mean() for o in expected[:2]).backward()
    for a,b in zip(inputs,ref_inputs):
        torch.testing.assert_close(a.grad,b.grad,atol=tolerance*.1,rtol=tolerance*3)
        assert torch.isfinite(a.grad).all()
    records.append(dict(dtype=str(dtype),forward_and_backward="PASS",
                        max_output_abs_error=max(float((a-b).abs().max()) for a,b in zip(actual,expected))))
Path("runs/fla_parity.json").write_text(json.dumps(dict(status="PASS",torch=torch.__version__,
    gpu=torch.cuda.get_device_name(),results=records),indent=2)+"\n")
print(json.dumps(records),flush=True)
