"""Archive results without canonical/raw data; optional large artifacts."""
import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--run-dir',required=True);p.add_argument('--mode',choices=['summary','full'],default='summary')
for name in ('logs','checkpoints','predictions'):p.add_argument('--include-'+name,action='store_true')
a=p.parse_args();root=Path(a.run_dir).resolve();out=root.parent/'archives';out.mkdir(exist_ok=True)
files=[]
for f in root.rglob('*'):
    if not f.is_file() or 'prepared' in f.relative_to(root).parts or 'attempts' in f.relative_to(root).parts:continue
    relative=f.relative_to(root)
    if f.suffix=='.ckpt' and not (a.include_checkpoints or a.mode=='full'):continue
    if f.name.startswith('predictions_') and f.name.endswith('.gz') and not (a.include_predictions or a.mode=='full'):continue
    if f.suffix=='.log' and not (a.include_logs or a.mode=='full'):continue
    files.append(f)
archive=out/f'yousician_fmkc_{root.name}_{a.mode}_{time.time_ns()}.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for f in files:tar.add(f,arcname=str(Path(root.name)/f.relative_to(root)))
h=hashlib.sha256()
with archive.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
archive.with_suffix(archive.suffix+'.sha256').write_text(h.hexdigest()+'  '+archive.name+'\n')
archive.with_suffix(archive.suffix+'.files.json').write_text(json.dumps([str(f.relative_to(root)) for f in files],indent=2))
print(archive)
print(f'Download after substituting your real host: scp <host>:{archive} .')
