"""Read the verified NDTK binary format without re-tokenizing or copying tokens."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

INDEX = np.dtype([("doc_key","<u8"),("offset","<u8"),("count","<u4"),
                  ("bytes","<u4"),("source","u1"),("tier","u1"),
                  ("flags","<u2"),("reserved","<u4")])


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def mapped(path, kind):
    path=Path(path)
    expected=b"NDTKTOK1" if kind=="tokens" else b"NDTKIDX1"
    dtype=np.dtype("<u2") if kind=="tokens" else INDEX
    with path.open("rb") as f:
        if f.read(8)!=expected:
            raise ValueError(f"Bad {kind} magic: {path}")
    if (path.stat().st_size-96)%dtype.itemsize:
        raise ValueError(f"Misaligned {kind} file: {path}")
    return np.memmap(path,mode="r",dtype=dtype,offset=96)


def prepare(root,output,budget_bytes=8_000_000_000,holdout=.03,seed=20260910):
    if budget_bytes<=0 or not 0<holdout<1:
        raise ValueError("Invalid data budget or holdout fraction")
    root,output=Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    main=json.loads((root/"MANIFEST.json").read_text())
    if main["status"]!="PASS" or main["vocab_size"]!=32000:
        raise ValueError("Unexpected source manifest")
    seal=main["dataset_archive_seal_sha256"]
    candidates=[]
    for p in sorted((root/"shards").glob("*.manifest.json")):
        m=json.loads(p.read_text())
        if m.get("dataset_archive_seal_sha256")==seal:
            candidates.append((p,m))
    rng=np.random.default_rng(seed)
    rng.shuffle(candidates)
    selected=[]
    total=0
    seen=set()
    for p,m in candidates:
        if m["status"]!="PASS" or m["vocab_sha256"]!=main["vocab_sha256"]:
            raise ValueError(f"Invalid shard {p}")
        ip=root/"shards"/m["index_file"]
        tp=root/"shards"/m["tokens_file"]
        if ip.stat().st_size!=m["index_file_bytes"] or tp.stat().st_size!=m["tokens_file_bytes"]:
            raise ValueError("File size mismatch")
        if digest(ip)!=m["index_sha256"]:
            raise ValueError("Index checksum mismatch")
        idx=mapped(ip,"index")
        take=[]
        for row in idx:
            if not int(row["flags"])&1:
                continue
            cost=2*int(row["count"])
            if cost==0 or total+cost>budget_bytes:
                continue
            doc=int(row["doc_key"])
            if doc in seen:
                raise ValueError("Duplicate document key")
            seen.add(doc)
            take.append((doc,int(row["offset"]),int(row["count"]),int(row["bytes"])))
            total+=cost
        if take:
            # Verify complete selected source shard once, not on every training run.
            if digest(tp)!=m["tokens_sha256"]:
                raise ValueError("Token checksum mismatch")
            selected.append((tp,ip,m,take))
        if budget_bytes-total<65536:
            break
    if not selected:
        raise ValueError("No eligible documents")
    records=[]
    sources=[]
    for shard,(tp,ip,m,rows) in enumerate(selected):
        sources.append(dict(tokens=str(tp),index=str(ip),tokens_sha256=m["tokens_sha256"],index_sha256=m["index_sha256"]))
        for doc,offset,count,nbytes in rows:
            # Stable document split independent of traversal/training RNG.
            h=int.from_bytes(hashlib.blake2b(f"{seed}:{doc}".encode(),digest_size=8).digest(),"little")
            records.append((shard,doc,offset,count,nbytes,int(h/2**64<holdout)))
    dtype=np.dtype([("shard","<u4"),("doc_key","<u8"),("offset","<u8"),
                    ("count","<u4"),("bytes","<u4"),("eval","u1")])
    docs=np.array(records,dtype=dtype)
    if not docs["eval"].any() or docs["eval"].all():
        raise ValueError("Subset must contain both train and evaluation documents")
    np.save(output/"documents.npy",docs,allow_pickle=False)
    manifest=dict(schema="rmala_subset_v1",source=str(root),vocab_size=main["vocab_size"],
                  vocab_sha256=main["vocab_sha256"],source_manifest_sha256=digest(root/"MANIFEST.json"),
                  seed=seed,budget_unit="decimal token-payload bytes, whole documents, excluding headers/index",
                  requested_bytes=budget_bytes,selected_bytes=total,documents=len(docs),
                  train_documents=int((docs["eval"]==0).sum()),eval_documents=int(docs["eval"].sum()),
                  train_tokens=int(docs["count"][docs["eval"]==0].sum()),
                  eval_tokens=int(docs["count"][docs["eval"]==1].sum()),
                  eval_text_utf8_bytes=int(docs["bytes"][docs["eval"]==1].sum()),
                  documents_sha256=digest(output/"documents.npy"),sources=sources)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


class Corpus:
    def __init__(self,directory):
        directory=Path(directory)
        self.manifest=json.loads((directory/"manifest.json").read_text())
        if digest(directory/"documents.npy")!=self.manifest["documents_sha256"]:
            raise ValueError("Subset document list changed")
        self.docs=np.load(directory/"documents.npy",mmap_mode="r",allow_pickle=False)
        self.tokens={}

    def document(self,row):
        shard=int(row["shard"])
        if shard not in self.tokens:
            self.tokens[shard]=mapped(self.manifest["sources"][shard]["tokens"],"tokens")
        offset,count=int(row["offset"]),int(row["count"])
        result=self.tokens[shard][offset:offset+count]
        if len(result)!=count or not len(result) or result[-1]!=2 or result.max()>=self.manifest["vocab_size"]:
            raise ValueError("Invalid document tokens, bounds, vocabulary or EOS")
        return result.astype(np.int64)

    def chunks(self,row,block):
        # EOS=2 provides a documented start context. Every document token is a
        # target exactly once. Chunks reset memory; chunks never mix documents.
        tok=self.document(row)
        for start in range(0,len(tok),block):
            target=tok[start:start+block]
            first=2 if start==0 else tok[start-1]
            inputs=np.concatenate(([first],target[:-1]))
            yield inputs,target


def metrics(total_nll,tokens,text_bytes):
    if tokens<=0 or text_bytes<=0:
        raise ValueError("Metrics require positive exact token and byte counts")
    return dict(bpb=total_nll/(math.log(2)*text_bytes),perplexity=math.exp(min(total_nll/tokens,700)),
                nll_nats=total_nll,tokens=tokens,text_utf8_bytes=text_bytes)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--budget-bytes",type=int,default=8_000_000_000)
    a=p.parse_args()
    print(json.dumps(prepare(a.root,a.output,a.budget_bytes),indent=2))
