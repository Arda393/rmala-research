"""Publish the validated allowlisted exports; token is read only from process env."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import logging
logging.set_verbosity_error()

ROOT=Path.cwd()
BASE=ROOT/'exports/hf_lm100_3b'
REPORT=ROOT/'reports/hf_publication_20260927.json'


def save(report):
    tmp=REPORT.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,indent=2)+'\n')
    tmp.replace(REPORT)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    token=os.environ['HF_TOKEN']
    api=HfApi(token=token)
    exported=json.loads((BASE/'export_report.json').read_text())
    validated=json.loads((BASE/'loader_validation.json').read_text())
    assert validated['status']=='PASS' and len(validated['variants'])==5
    report=json.loads(REPORT.read_text()) if REPORT.exists() else dict(repositories={},status='running')
    failures=[]
    for model in exported:
        folder=Path(model['path'])
        sums=json.loads((folder/'SHA256SUMS.json').read_text())
        # Only files in this explicit export manifest may leave the project.
        assert sha(folder/'model.safetensors')==model['safetensors_sha256']
        for name,expected in sums.items():
            p=folder/name
            assert p.resolve().is_relative_to(folder.resolve()) and p.is_file()
            assert sha(p)==expected,name
            if p.suffix in ['.py','.md','.json','.txt']:
                content=p.read_text(encoding='utf-8')
                assert token not in content,'Credential unexpectedly present in export'
                assert not re.search(r'\bhf_[A-Za-z0-9]{20,}\b',content),'Credential-like text in export'
        for org in ['Ethosoft','MercanAI']:
            repo=f'{org}/{model["name"]}'
            entry=report['repositories'].setdefault(repo,dict(status='planned',variant=model['variant']))
            try:
                if entry['status']=='verified':
                    continue
                if entry['status']=='planned':
                    # Refuse overwriting an unrelated pre-existing repository.
                    api.create_repo(repo_id=repo,repo_type='model',private=False,exist_ok=False)
                    entry['status']='created';save(report)
                print(json.dumps(dict(stage='uploading',repo=repo,bytes=model['bytes'])),flush=True)
                commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=str(folder),
                    allow_patterns=list(sums)+['SHA256SUMS.json'],
                    commit_message='Publish exact final 100M / 3B-token research checkpoint and reproducibility files')
                entry.update(status='uploaded',commit=commit.oid,url=f'https://huggingface.co/{repo}')
                save(report)
                info=api.model_info(repo_id=repo,files_metadata=True)
                assert info.private is False
                files={f.rfilename:f for f in info.siblings}
                assert set(sums).issubset(files)
                weight=files['model.safetensors']
                assert weight.size==model['bytes']
                assert weight.lfs.sha256==model['safetensors_sha256']
                for filename in ['README.md','config.json','inference.py','surface-vocab.bin','SHA256SUMS.json']:
                    downloaded=hf_hub_download(repo_id=repo,filename=filename,revision=commit.oid,token=token)
                    assert sha(downloaded)==sha(folder/filename),filename
                entry.update(status='verified',weights_sha256=weight.lfs.sha256,bytes=weight.size,files=len(files))
                save(report)
                print(json.dumps(dict(stage='verified',repo=repo,commit=commit.oid,url=entry['url'])),flush=True)
            except Exception as error:
                response=getattr(error,'response',None)
                failure=dict(repo=repo,type=type(error).__name__,http_status=getattr(response,'status_code',None))
                failures.append(failure)
                entry['last_error']=failure
                save(report)
                # No exception strings, headers, token values, or signed upload URLs in logs.
                print(json.dumps(dict(stage='failed',**failure)),flush=True)
    report.update(status='completed' if not failures else 'partial',failures=failures)
    save(report)
    print(json.dumps(dict(status=report['status'],verified=sum(x['status']=='verified' for x in report['repositories'].values()))),flush=True)
    return 0 if not failures else 1


if __name__=='__main__':sys.exit(main())
