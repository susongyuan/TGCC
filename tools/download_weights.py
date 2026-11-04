"""Download pinned hotdog v3fast and its TC/base dependencies; no GPU work."""
import argparse,json
from pathlib import Path
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('weights'));a=p.parse_args()
r=Path(__file__).resolve().parents[1];pins=json.loads((r/'reproduction/releases.json').read_text())
for repo,directory in [('Songyuan666/TGCC-hotdog-v3fast-step02153','hotdog-v3fast'),('Songyuan666/TC-pi0.5','tc-base')]:
 snapshot_download(repo,revision=pins[repo]['revision'],local_dir=a.output/directory)
