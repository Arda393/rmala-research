"""Lossless deduplicated candidate archive; no pickle or numeric rounding."""
import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path


def pack(root,destination):
    root=Path(root);records=[];blobs={}
    def encode(values,kind):
        b=struct.pack('<'+kind*len(values),*values)
        assert list(struct.unpack('<'+kind*len(values),b))==values
        h=hashlib.sha256(b).hexdigest();blobs[h]=b
        return dict(hash=h,kind=kind,count=len(values))
    for line in (root/'candidates.jsonl').open():
        c=json.loads(line)
        for field in ['base','candidate']:c[field]=encode(c[field],'f')
        probs=c['probability'];ints=[round(x*100000) for x in probs]
        assert [x/100000 for x in ints]==probs
        c['probability']=encode(ints,'i');c['distance']=encode(c['distance'],'i')
        c['groups']={n:encode(m,'?') for n,m in c['groups'].items()}
        records.append(c)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_LZMA) as z:
        for p in sorted(root.iterdir()):
            if p.is_file() and p.name!='candidates.jsonl':z.write(p,'runs/'+root.name+'/'+p.name)
        z.writestr('runs/'+root.name+'/candidates_packed.json',json.dumps(records))
        for h,b in blobs.items():z.writestr('arrays/'+h,b)
        z.write(Path(__file__),'pack_v9_candidates.py')
    return len(blobs)


def unpack(extracted,run_name):
    root=Path(extracted);out=root/'runs'/run_name
    def decode(d):
        b=(root/'arrays'/d['hash']).read_bytes();assert hashlib.sha256(b).hexdigest()==d['hash']
        return list(struct.unpack('<'+d['kind']*d['count'],b))
    with (out/'candidates.jsonl').open('w') as f:
        for c in json.loads((out/'candidates_packed.json').read_text()):
            for field in ['base','candidate','distance']:c[field]=decode(c[field])
            c['probability']=[x/100000 for x in decode(c['probability'])]
            c['groups']={n:decode(d) for n,d in c['groups'].items()}
            f.write(json.dumps(c)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('destination');a=p.parse_args();print(pack(a.root,a.destination))
