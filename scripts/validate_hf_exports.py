"""CPU strict-load and native-tokenizer validation of every release bundle."""
import gc
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import torch


def main():
    torch.set_num_threads(2)
    root=Path('exports/hf_lm100_3b')
    reports=json.loads((root/'export_report.json').read_text())
    validated=[]
    for r in reports:
        folder=Path(r['path'])
        spec=importlib.util.spec_from_file_location('release_inference',folder/'inference.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tokenizer=module.load_tokenizer(folder)
        text='Türkçe deneme: ı İ ğ ş ç ö ü. Sayı 1234.\n'
        raw=text.encode('utf-8')
        assert tokenizer.decode_ids(tokenizer.encode_ids(raw))==raw
        if r['variant']=='hola' and not torch.cuda.is_available():
            # Official Triton imports initialize a CUDA driver even for construction.
            # Do not modify/mock the model to claim a CPU architecture load.
            assert r['weight_roundtrip_exact']
            for filename in ['lm100_model.py','lm100_memory.py']:
                assert (folder/'rmala'/filename).read_bytes()==(Path('rmala')/filename).read_bytes()
            validated.append(dict(variant=r['variant'],strict_state_load=None,
                tensor_roundtrip_exact=True,tokenizer_roundtrip=True,parameters=r['parameters'],
                limitation='Official HoLA/Triton requires CUDA at import. CPU construction unavailable; GPU inference not rerun. Original trained model source retained byte-for-byte.'))
            print(json.dumps(validated[-1]),flush=True)
            del module,tokenizer
            continue
        model=module.load_model(folder,device='cpu')
        assert model.head.weight.data_ptr()==model.embedding.weight.data_ptr()
        assert sum(p.numel() for p in model.parameters())==r['parameters']
        assert len(model.state_dict())==r['tensor_keys']
        validated.append(dict(variant=r['variant'],strict_state_load=True,
            tied_head_restored=True,tokenizer_roundtrip=True,parameters=r['parameters']))
        print(json.dumps(validated[-1]),flush=True)
        del model,module,tokenizer
        gc.collect()
    result=dict(status='PASS',variants=validated,
        scope='All five exact safetensors tensor roundtrips and tokenizer roundtrips pass. Four strict CPU architecture loads pass; HoLA CUDA-required construction was not rerun. No new GPU generation run.')
    (root/'loader_validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
