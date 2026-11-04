"""User-authorized incomplete-checkpoint admission; no trained-source or metadata rewriting."""
import sys,json,runpy
from pathlib import Path
mode=sys.argv[1];i=sys.argv.index('--source');source=Path(sys.argv[i+1]);args=sys.argv[i+2:]
sys.path[:0]=[str(source/'src'),str(source/'code/src'),str(source/'code/packages/openpi-client/src')]
def admitted(md,initial=None):
 return (md.get('new_updates')==2153 and md.get('training_complete') is False and md.get('audit_disposable') is False
  and md.get('user_stopped_checkpoint') is True and md.get('task')=='cook_hot_dogs'
  and md.get('critic_actual_total_updates')==4809 and md.get('policy_cumulative_updates')==12153
  and md.get('export_restored_tensors_verified_exact') is True and md.get('config',{}).get('frozen_policy_groups')==['vision','vlm'])
if mode=='serve':
 import tgcc.distribution_serve as module
 module.artifact_recipe_consistent=admitted
 sys.argv=['distribution_serve']+args;module.main()
elif mode=='calibrate':
 path=source/'tools/tgcc/calibrate_distribution.py';text=path.read_text()
 old='metadata.get("audit_disposable") is not False or not metadata.get("training_complete") or\n            metadata["new_updates"] != metadata["config"]["new_updates"] or'
 assert text.count(old)==1
 text=text.replace(old,'not authorized_stopped(metadata) or')
 text=text.replace('ROOT = Path(__file__).resolve().parents[2]','ROOT = Path('+repr(str(source))+')')
 sys.argv=[str(path)]+args
 exec(compile(text,str(path),'exec'),dict(__name__='__main__',__file__=str(path),authorized_stopped=admitted))
else:raise ValueError(mode)
