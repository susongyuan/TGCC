"""Verify exact source and all downloaded inference assets without CUDA allocation."""
from pathlib import Path
import argparse,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def verify(assets=None):
 source=json.loads((ROOT/'reproduction/frozen_source_sha256.json').read_text())
 for n,w in source.items():assert sha(ROOT/'frozen'/n)==w,n
 if assets:
  md=json.loads((assets/'MANIFEST.json').read_text())
  for n,v in md['files'].items():assert sha(assets/n)==v['sha256'],n
 return dict(source_files_verified=len(source),asset_files_verified=len(md['files']) if assets else 0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--assets',type=Path);a=p.parse_args();print(json.dumps(verify(a.assets)))
