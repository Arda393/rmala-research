import copy
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from rmala.lm100_data import HEADER, TokenStream
from rmala.lm100_train import training_batch, evaluation_chunks, save_checkpoint


class RunnerTests(unittest.TestCase):
    def make_stream(self,root,ids):
        (root/'shards').mkdir()
        raw=struct.pack('<'+'H'*len(ids),*ids)
        (root/'shards/a').write_bytes(bytes(HEADER)+raw)
        manifest=dict(dataset_root=str(root),train_segments=[dict(tokens_file='a',take_tokens=len(ids))])
        return TokenStream(manifest),raw

    def test_exact_targets_partial_batch_resume_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream,raw=self.make_stream(Path(tmp),list(range(3,31)))
            targets=[]
            payload=b''
            for cursor,n in [(0,12),(12,12),(24,3)]:
                x,y,block=training_batch(stream,cursor,n,4,device='cpu')
                self.assertEqual(int((y!=-100).sum()),n)
                self.assertEqual(x.numel(),((n+3)//4)*4)
                targets.extend(y[y!=-100].tolist())
                payload+=block
            self.assertEqual(targets,list(range(4,31)))
            self.assertEqual(payload,raw[2:])
            self.assertEqual(hashlib.sha256(payload).hexdigest(),hashlib.sha256(stream.read(1,27)).hexdigest())

    def test_complete_document_bpb_chunk_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stream,raw=self.make_stream(root,[5,2,9,7,8,10,2])
            manifest=stream.manifest
            manifest['eval_start_token']=2
            manifest['test']=dict(documents=[dict(shard='a',offset=0,tokens=7,text_utf8_bytes=19,
                                                  token_sha256=hashlib.sha256(raw).hexdigest())])
            chunks=list(evaluation_chunks(manifest,'test',3))
            self.assertEqual([x[0].tolist() for x in chunks],[[2,5,2],[9,7,8],[10]])
            self.assertEqual(np.concatenate([x[1] for x in chunks]).tolist(),[5,2,9,7,8,10,2])
            self.assertEqual(sum(x[3] for x in chunks),19)
            self.assertEqual(sum(int(x[2].sum()) for x in chunks),6)
            # The embedded ID=2 is content; only the terminal EOS is excluded.
            self.assertTrue(chunks[0][2][1])

    def test_checkpoint_restores_optimizer_and_cursor(self):
        torch.manual_seed(2)
        model=torch.nn.Linear(3,2)
        opt=torch.optim.AdamW(model.parameters(),lr=.01)
        x=torch.randn(7,3)
        def step(m,o):
            o.zero_grad()
            m(x).square().mean().backward()
            o.step()
        step(model,opt)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            state=dict(cursor=12,updates=1,target_chain_sha256='a'*64)
            save_checkpoint(out,model,opt,state)
            step(model,opt)
            uninterrupted=copy.deepcopy(model.state_dict())
            checkpoint=torch.load(out/'latest.pt',weights_only=False)
            fresh=torch.nn.Linear(3,2)
            fresh_opt=torch.optim.AdamW(fresh.parameters(),lr=.01)
            fresh.load_state_dict(checkpoint['model'])
            fresh_opt.load_state_dict(checkpoint['optimizer'])
            step(fresh,fresh_opt)
            for k,v in fresh.state_dict().items():
                torch.testing.assert_close(v,uninterrupted[k],atol=0,rtol=0)
            self.assertEqual(checkpoint['state'],state)
            save_checkpoint(out,fresh,fresh_opt,dict(cursor=24))
            self.assertTrue((out/'previous.pt').exists())
            self.assertFalse((out/'checkpoint.tmp').exists())


if __name__=='__main__':
    unittest.main()
