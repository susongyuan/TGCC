"""Portable launcher; preserves frozen source, checkpoint metadata and native action limits."""
import argparse,os,sys,runpy
from pathlib import Path
from verify import verify
p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--base-assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=18835);a=p.parse_args()
r=Path(__file__).resolve().parents[1];a.assets=a.assets.resolve();a.base_assets=a.base_assets.resolve();verify(a.assets)
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
source=r/'frozen'
sys.path[:0]=[str(source/'src'),str(source/'code/src'),str(source/'code/packages/openpi-client/src')]
# Route the original public tokenizer URI to the checksum-bound mirrored file.
from openpi.shared import download
original=download.maybe_download
from hashlib import sha256
tokenizer=a.base_assets/'common/paligemma_tokenizer.model'
manifest=__import__('json').loads((a.base_assets/'release_manifest.json').read_text())
# The base release verifier remains available from the original TC repository.
wanted=next(v['sha256'] for v in manifest['files'] if v['path']=='common/paligemma_tokenizer.model')
assert tokenizer.is_file() and sha256(tokenizer.read_bytes()).hexdigest()==wanted,'Tokenizer checksum differs'
def local_tokenizer(url,*args,**kwargs):
 if str(url)=='gs://big_vision/paligemma_tokenizer.model':return tokenizer
 return original(url,*args,**kwargs)
download.maybe_download=local_tokenizer
sys.argv=['stopped_checkpoint_entry','serve','--source',str(source),
 '--checkpoint',str(a.base_assets/'pi0.5-behaviour-base'),'--parent',str(a.base_assets/'cook_hot_dogs/step5000'),
 '--distribution',str(a.assets/'checkpoint'),'--retrieval-data',str(a.assets/'retrieval_library'),
 '--plan',str(source/'reports/data_plan_vocabulary.json'),'--protocol',str(source/'configs/tgcc/evaluation_pairing.json'),
 '--output',str(a.output),'--port',str(a.port),'--oracle-upper-bound']
runpy.run_path(str(r/'runtime_extensions/stopped_checkpoint_entry.py'),run_name='__main__')
