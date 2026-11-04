"""Install the exact checkpoint-bound Transformer replacements into an isolated venv."""
import sys,shutil
from pathlib import Path
import transformers
assert sys.prefix!=sys.base_prefix,'Use a virtual environment'
assert transformers.__version__=='4.53.2'
r=Path(__file__).resolve().parents[1];source=r/'frozen/code/src/openpi/models_pytorch/transformers_replace';target=Path(transformers.__file__).parent
files=list(source.rglob('*.py'));assert len(files)==5
for p in files:
 q=target/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
print('5 pinned Transformer files installed')
