"""Capture precisely which project source/config files are present for a run."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
records={}
for directory in ('rmala','scripts','tests','configs'):
    for path in sorted((root/directory).rglob('*')):
        if path.is_file() and path.suffix in ('.py','.json'):
            records[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
(root/'runs').mkdir(exist_ok=True)
(root/'runs'/'source_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
