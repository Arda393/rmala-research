import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rmala.data import INDEX,Corpus,prepare,digest


class DataTests(unittest.TestCase):
    def test_headers_document_split_and_exact_byte_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"source";shards=root/"shards";shards.mkdir(parents=True)
            rows=[];tokens=[]
            for i in range(100):
                rows.append((i,len(tokens),4,9,0,0,1,0));tokens.extend([10,11,12,2])
            tp=shards/"test.tokens.bin";ip=shards/"test.index.bin"
            tp.write_bytes(b"NDTKTOK1"+bytes(88)+np.array(tokens,dtype="<u2").tobytes())
            ip.write_bytes(b"NDTKIDX1"+bytes(88)+np.array(rows,dtype=INDEX).tobytes())
            common=dict(status="PASS",vocab_size=32000,vocab_sha256="test",dataset_archive_seal_sha256="seal")
            (root/"MANIFEST.json").write_text(json.dumps(common))
            sm=dict(common,index_file=ip.name,tokens_file=tp.name,index_file_bytes=ip.stat().st_size,
                    tokens_file_bytes=tp.stat().st_size,index_sha256=digest(ip),tokens_sha256=digest(tp))
            (shards/"test.manifest.json").write_text(json.dumps(sm))
            # Another source's malformed manifest must never enter the selected corpus.
            (shards/"other.manifest.json").write_text(json.dumps({"dataset_archive_seal_sha256":"other"}))
            output=Path(temp)/"subset"
            m=prepare(root,output,budget_bytes=800,holdout=.2)
            self.assertEqual(m["selected_bytes"],800)
            c=Corpus(output)
            train=set(c.docs["doc_key"][c.docs["eval"]==0]);ev=set(c.docs["doc_key"][c.docs["eval"]==1])
            self.assertFalse(train&ev)
            self.assertEqual(len(train|ev),100)
            chunks=list(c.chunks(c.docs[0],3))
            self.assertEqual([int(v) for _,y in chunks for v in y],[10,11,12,2])
            self.assertEqual(int(chunks[0][0][0]),2)
            self.assertEqual(m["eval_text_utf8_bytes"],9*len(ev))


if __name__=="__main__":
    unittest.main()
