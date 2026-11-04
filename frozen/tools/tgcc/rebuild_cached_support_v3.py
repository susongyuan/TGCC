"""Rebuild corrected support from immutable real frozen-encoder raw caches.

No model forward, simulator, policy updates or fabricated observations. The
old extraction provenance is retained separately from this feature transform.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from tgcc.transition_features import (FEATURE_SCHEMA,EXECUTION_TARGET,stable_scoring_context,
    current_outcome_from_context,outcome_mask,transition_target,fit_transition_statistics,normalize_transition)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();old=a.source.resolve();out=a.output.resolve()
    if old==out or out.exists():raise ValueError('Independent fresh output directory required')
    identity=json.loads((old/'identity.json').read_text())
    if identity.get('schema')!='tgcc.normal_support_library.v2' or identity.get('encoder_frozen') is not True:
        raise ValueError('Only audited frozen-encoder v2 real raw support can be transformed')
    for name,expected in identity['artifact_sha256'].items():
        if sha(old/name)!=expected:raise ValueError('Original support artifact changed: '+name)
    for name in ('vision_encoder','vision_config'):
        file=old/(name+('.safetensors' if name=='vision_encoder' else '.json'))
        if sha(file)!=identity[name+'_sha256']:raise ValueError('Original frozen encoder changed')
    rows=[json.loads(s) for s in (old/'rows.jsonl').read_text().splitlines()]
    lookup={(str(r['episode_id']),r['sequence_steps'][0]):i for i,r in enumerate(rows)}
    caches={};records={};input_hashes={}
    for file in sorted(old.glob('raw_episode_*.npz')):
        record_file=file.with_suffix('.rows.json');meta=json.loads(record_file.read_text())
        with np.load(file,allow_pickle=False) as data:values={key:data[key] for key in data.files}
        if any(len(value)!=len(meta) for value in values.values()):raise ValueError('Partial raw cache')
        for j,row in enumerate(meta):
            key=(str(row['episode_id']),row['sequence_steps'][0])
            if key not in lookup or key in records:raise ValueError('Repeated/foreign raw source anchor')
            original=rows[lookup[key]]
            if row!=original:raise ValueError('Raw source provenance mismatch')
            records[key]=row;caches[key]={name:value[j] for name,value in values.items()}
        input_hashes[file.name]=sha(file);input_hashes[record_file.name]=sha(record_file)
    if len(records)!=len(rows):raise ValueError('Incomplete original real support')
    ordered=[caches[(str(r['episode_id']),r['sequence_steps'][0])] for r in rows]
    with np.load(old/'codec.npz',allow_pickle=False) as data:
        visual_mean=data['visual_mean'];components=data['visual_components']
    width=len(visual_mean);v=components.shape[1]
    project=lambda x:(np.asarray(x,np.float32)-visual_mean)@components
    contexts=[];changes=[];masks=[]
    for record in ordered:
        raw=record['raw_context'];context=np.r_[project(raw[:width]),raw[width:]]
        context=stable_scoring_context(context,v)
        before=current_outcome_from_context(context,v)
        rawafter=record['raw_after_features'];after=np.r_[project(rawafter[:width]),rawafter[width:]]
        amask=np.r_[np.ones(v,bool),record['raw_after_feature_mask'][width:]] & outcome_mask(after,v)
        change,known=transition_target(before,after,outcome_mask(before,v),amask)
        contexts.append(context);changes.append(change);masks.append(known)
    contexts=np.asarray(contexts,np.float32);changes=np.asarray(changes,np.float32);masks=np.asarray(masks,bool)
    train=np.asarray([i for i,r in enumerate(rows) if r['split']=='train']);validation=np.asarray([i for i,r in enumerate(rows) if r['split']=='validation'])
    mean=contexts[train].mean(0);std=np.maximum(contexts[train].std(0),.01)
    after_mean,after_std,supported=fit_transition_statistics(changes,masks,train)
    normalized,masks=normalize_transition(changes,masks,dict(after_mean=after_mean,after_std=after_std,after_supported=supported))
    arrays={name:np.stack([r[name] for r in ordered]) for name in
            ('actions','action_mask','actual_native','next_index','next_mask','reference_state','reference_after_state')}
    arrays.update(context=((contexts-mean)/std).astype(np.float32),after_features=normalized,
        after_feature_mask=masks,reference_visual_context=project(np.stack([r['raw_visual'] for r in ordered])).astype(np.float32))
    tmp=out.with_name(out.name+'.building')
    if tmp.exists():raise ValueError('Retained partial rebuild requires inspection')
    tmp.mkdir(parents=True)
    np.savez(tmp/'codec.npz',visual_mean=visual_mean,visual_components=components,context_mean=mean,context_std=std,
        after_mean=after_mean,after_std=after_std,after_supported=supported)
    np.savez_compressed(tmp/'samples.npz',**arrays)
    shutil.copy2(old/'rows.jsonl',tmp/'rows.jsonl')
    for name in ('vision_encoder.safetensors','vision_config.json'):(tmp/name).symlink_to(old/name)
    hashes={name:sha(tmp/name) for name in ('samples.npz','rows.jsonl','codec.npz')}
    new=dict(identity,schema='tgcc.normal_support_library.v3',feature_schema=FEATURE_SCHEMA,execution_target=EXECUTION_TARGET,
        artifact_sha256=hashes,feature_artifact_sha256=hashes,codec_sha256=hashes['codec.npz'],feature_codec_sha256=hashes['codec.npz'],
        retrieval_index_sha256=hashlib.sha256((hashes['samples.npz']+hashes['rows.jsonl']).encode()).hexdigest(),
        raw_cache_source_library_sha256=sha(old/'identity.json'),raw_cache_source_hashes=input_hashes,
        rebuild_transform_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            (Path(__file__),ROOT/'src/tgcc/transition_features.py')},
        feature_rebuild_scope='real_raw_caches; frozen_vision_and_train_only_PCA_unchanged; corrected_current_features_and_actual_deltas')
    (tmp/'identity.json').write_text(json.dumps(new,indent=2)+'\n')
    receipt=dict(status='completed_real_cached_feature_rebuild',source=str(old),output=str(out),
        train_samples=len(train),validation_samples=len(validation),source_episodes=len({r['episode_id'] for r in rows}),
        goal_confidence_context_column_zero=bool(np.all(arrays['context'][:,968]==0)),
        goal_confidence_not_execution_target=bool(not masks[:,921].any()),
        finite_contexts=bool(np.isfinite(arrays['context']).all()),finite_observed_deltas=bool(np.isfinite(normalized[masks]).all()),
        added_regular_file_bytes=sum(p.stat().st_size for p in tmp.iterdir() if p.is_file() and not p.is_symlink()),
        shared_frozen_encoder=True,model_forwards=0,training_updates=0,robot_actions=0)
    if not all(receipt[k] for k in ('goal_confidence_context_column_zero','goal_confidence_not_execution_target','finite_contexts','finite_observed_deltas')):
        raise ValueError('Corrected real feature invariants failed')
    (tmp/'rebuild_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    os.replace(tmp,out);print(json.dumps(receipt))


if __name__=='__main__':main()
