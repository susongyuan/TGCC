"""Build owned ranking pairs from genuine synchronized collection.

500 pairs is 1,000 candidate records, not 500 independent experiments. Control
admissibility, preserved-grasp ranking, and planning preconditions are reported
separately; real local-contract outcomes are additional evidence, not fabricated
counterfactual after-states. The command refuses to fill missing task examples.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from tgcc.contracts import Evidence, Predicate, TaskContract, Truth, build_contract
from tgcc.evaluation_data import (contract_record, held_by_side, maintain_contract,
                                   validate_pair, verify_transition)
from tgcc.runtime import snapshot_evidence
from tgcc.robot import action_spec


def matched_pre(contract,evidence,now):
    values=[evidence.get(p.key,Evidence()).truth_for(p,now)
            for p in contract.predicates if p.category=='pre' and p.required_at_current_stage]
    return bool(values) and all(v!=Truth.UNKNOWN for v in values) and all(
        (evidence[p.key].truth_for(p,now)==Truth.TRUE)==p.expected
        for p in contract.predicates if p.category=='pre' and p.required_at_current_stage)


def failed_pre(contract,evidence,now):
    return [p for p in contract.predicates if p.category=='pre' and p.required_at_current_stage
            and evidence.get(p.key,Evidence()).truth_for(p,now)!=Truth.UNKNOWN
            and (evidence[p.key].truth_for(p,now)==Truth.TRUE)!=p.expected]


def candidates(directory,output_root,spec):
    directory=Path(directory).resolve()
    report=json.loads((directory/'report.json').read_text())
    if report['status']!='completed' or not report.get('source_episode_id'):
        raise ValueError('Only completed source-identified collections are admitted')
    if not report.get('rgb_recorded') or not report['restore_check']['verified']:
        raise ValueError('Incomplete visual collection or restore')
    source=report['source_episode_id'];split=report['source_split']
    artifact=json.loads((ROOT/'reports/data_plan_vocabulary.json').read_text())
    contracts=[build_contract(c) for c in artifact['candidates']]
    rows=[]
    for file in sorted(directory.glob('graphs_*/transitions.jsonl')):
        transitions=[json.loads(s) for s in file.read_text().splitlines() if s]
        for i,row in enumerate(transitions):
            before,after,actual=verify_transition(file.parent,row)
            for r in (before,after):
                if not r['raw'].get('depth') or not r['raw'].get('camera_calibration'):
                    raise ValueError('RGB-D/calibration are mandatory for this collection')
                if r['raw']['collection_provenance']['source_episode_id']!=source:
                    raise ValueError('Collector/source episode mismatch')
            annotations=row.get('execution_annotations')
            if annotations is None or len(annotations)!=len(actual):
                raise ValueError('Missing exact actual-action provenance')
            if any(a.get('source_episode_id')!=source for a in annotations):
                raise ValueError('Mixed source actions')
            source_steps=[a['source_step'] for a in annotations]
            if source_steps!=list(range(source_steps[0],source_steps[0]+len(actual))):
                raise ValueError('Nonconsecutive source prefix')
            context=f'{source}:{before["session"]}:{source_steps[0]}'
            ev=snapshot_evidence(before['raw']['graph_snapshot'],before['timestamp'])
            sides=held_by_side(before)
            kept=[name for (name,side),value in sides.items() if value.truth(before['timestamp'])==Truth.TRUE
                    and np.all(np.asarray([a['original_native'] for a in annotations])[:,14 if side=='left' else 22]<-.5)]
            contract=maintain_contract(kept) if kept else TaskContract(-1,'observe_current_context',(
                Predicate('observed','hotdog.n.02_1',category='pre'),))
            original=np.asarray([a['original_native'] for a in annotations],np.float32)
            original_padded=np.zeros((32,23),np.float32);original_padded[:len(original)]=original
            actual_padded=np.zeros((32,23),np.float32);actual_padded[:len(actual)]=actual
            base=dict(source_kind='collection',source_episode_id=source,split=split,
                directory=str(file.parent.relative_to(output_root)),transition=row,
                history=transitions[max(0,i-16):i],plan_sha256=before['plan_sha256'],
                independent_context_id=context,source_sha256=report['source_actions_sha256'],
                quality_certificate=dict(actual_time_alignment=True,depth_recorded=True,
                    no_counterfactual_after=True,policy_imitation_allowed=False))
            # Controlled failure rows supply real retrospective supervision and
            # a closed reference versus the actual release candidate.
            for (name,side),value in sides.items():
                d=14 if side=='left' else 22
                if value.truth(before['timestamp'])!=Truth.TRUE or not np.all(original[:,d]<-.5):continue
                neg=original_padded.copy();neg[:len(actual),d]=1
                sup=dict(rank_valid=True,reference_native=original_padded.tolist(),negative_native=neg.tolist(),
                    negative_kind='task',certificate=dict(rule='preserve_current_verified_grasp',object_id=name,
                        side=side,dimension=d,source='actual_pre_action_grasp_feedback'),
                    contract=contract_record(maintain_contract([name])),execution_scope='local_contract',
                    policy_imitation_allowed=False,synthetic_after_state=None)
                validate_pair(sup,spec,before,actual,row['action_mask'])
                rows.append(dict(base,id=f'{context}:task:{side}:{name}',body_group=side,supervision=sup))
            # Failure actions are never silently promoted as a good reference.
            if any(a['is_controlled_release'] for a in annotations):continue
            for body,dimensions in [('base',range(3)),('left',range(7,14)),('right',range(15,22))]:
                schedule=i+int(source)
                d=list(dimensions)[schedule%len(dimensions)]
                upper=(schedule//len(dimensions))%2==0
                impulse=(schedule//(2*len(dimensions)))%2==0
                step=schedule%len(actual)
                modified=[step] if impulse else list(range(len(actual)))
                fraction=(0.002,0.01,0.05,0.15)[(schedule//7)%4]
                margin=max(1e-3,fraction*(spec.upper[d]-spec.lower[d]))
                value=np.float32(spec.upper[d]+margin if upper else spec.lower[d]-margin)
                neg=actual_padded.copy();neg[modified,d]=value
                sup=dict(rank_valid=True,reference_native=actual_padded.tolist(),negative_native=neg.tolist(),
                    negative_kind='physical',certificate=dict(rule='native_controller_limit',dimension=d,
                        dimension_name=spec.names[d],bound_side='upper' if upper else 'lower',
                        bound=float(spec.upper[d] if upper else spec.lower[d]),candidate_value=float(value),
                        margin=float(margin),modified_steps=modified,subtype='impulse' if impulse else 'sustained'),
                    contract=contract_record(contract),execution_scope='local_contract',
                    policy_imitation_allowed=False,synthetic_after_state=None)
                validate_pair(sup,spec,before,actual,row['action_mask'])
                rows.append(dict(base,id=f'{context}:physical:{body}',body_group=body,supervision=sup))
            positive=[c for c in contracts if matched_pre(c,ev,before['timestamp'])]
            negative=[(c,failed_pre(c,ev,before['timestamp'])) for c in contracts]
            negative=[(c,failed) for c,failed in negative if failed]
            if positive and negative:
                # This certifies prerequisite legality only, not completion of
                # an unexecuted alternative plan or a task-level failure.
                good=positive[i%len(positive)];bad,false=negative[i%len(negative)]
                good=TaskContract(good.skill,good.candidate_id,tuple(p for p in good.predicates if p.category=='pre'))
                bad=TaskContract(bad.skill,bad.candidate_id,tuple(p for p in bad.predicates if p.category=='pre'))
                body=('base','left','right')[i%3]
                sup=dict(rank_valid=True,reference_native=actual_padded.tolist(),negative_native=actual_padded.tolist(),
                    negative_kind='planning',certificate=dict(rule='known_false_subtask_precondition',
                        false_predicates=[list(p.key) for p in false],scope='precondition_legality_only'),
                    contract=contract_record(good),negative_contract=contract_record(bad),
                    execution_scope='local_contract',policy_imitation_allowed=False,synthetic_after_state=None)
                validate_pair(sup,spec,before,actual,row['action_mask'])
                rows.append(dict(base,id=f'{context}:planning',body_group=body,supervision=sup))
    return rows


def select(rows,split,count,min_task):
    pools=defaultdict(list)
    for r in rows:
        if r['split']==split:pools[r['body_group'],r['supervision']['negative_kind']].append(r)
    # Spread each pool across episodes and across its entire temporal span.
    for key,values in pools.items():
        episodes=defaultdict(list)
        for v in values:episodes[v['source_episode_id']].append(v)
        for v in episodes.values():v.sort(key=lambda r:r['transition']['before']['timestamp'])
        spread=[]
        for episode,values in sorted(episodes.items()):
            order=np.argsort([int.from_bytes(hashlib.sha256(v['id'].encode()).digest()[:8],'big') for v in values])
            episodes[episode]=[values[i] for i in order]
        while any(episodes.values()):
            for values in episodes.values():
                if values:spread.append(values.pop())
        pools[key]=spread
    result=[];used=Counter();contexts=defaultdict(Counter)
    target={'base':(count+2)//3,'left':(count+1)//3,'right':count//3}
    # Guarantee real grasp-based task candidates before adding easier control
    # negatives. No fallback is allowed when genuine evidence is absent.
    task_available=sum(len(pools[b,'task']) for b in ('left','right'))
    if task_available<min_task:raise ValueError(f'{split} only {task_available} verified task pairs; need {min_task}')
    kinds=('task','planning','physical')
    task_selected=0
    while len(result)<count:
        progressed=False
        for kind in kinds:
            for body in ('base','left','right'):
                if used[body]>=target[body]:continue
                pool=pools[body,kind]
                while pool:
                    row=pool.pop(0);context=row['independent_context_id']
                    if contexts[context][kind] or sum(contexts[context].values())>=2:continue
                    contexts[context][kind]+=1;used[body]+=1;result.append(row)
                    task_selected+=kind=='task';progressed=True;break
        if not progressed:raise ValueError(f'Insufficient unique certified contexts for {split}: {len(result)}/{count}')
    if task_selected<min_task:raise ValueError('Selection failed its grasp pair floor')
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--collections',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--train-pairs',type=int,default=400)
    p.add_argument('--validation-pairs',type=int,default=100)
    p.add_argument('--min-train-task',type=int,default=40)
    p.add_argument('--min-validation-task',type=int,default=10)
    a=p.parse_args();root=a.data_root.resolve();spec=action_spec(ROOT)
    all_rows=[row for path in a.collections for row in candidates(path,root,spec)]
    chosen=select(all_rows,'train',a.train_pairs,a.min_train_task)+select(all_rows,'validation',a.validation_pairs,a.min_validation_task)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    with a.output.open('x') as f:
        for row in chosen:f.write(json.dumps(row,separators=(',',':'))+'\n')
    counts=Counter((r['split'],r['body_group'],r['supervision']['negative_kind']) for r in chosen)
    report=dict(schema='tgcc.evaluation_pairs.v2',pairs=len(chosen),candidate_records=len(chosen)*2,
        available_by_split_kind={str(k):v for k,v in Counter((r['split'],r['supervision']['negative_kind']) for r in all_rows).items()},
        selected_by_split_body_kind={str(k):v for k,v in counts.items()},
        independent_source_episodes=len({r['source_episode_id'] for r in chosen}),
        distinct_pre_action_contexts=len({r['independent_context_id'] for r in chosen}),
        train_episodes=sorted({r['source_episode_id'] for r in chosen if r['split']=='train'}),
        validation_episodes=sorted({r['source_episode_id'] for r in chosen if r['split']=='validation'}),
        manifest_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        outcome_scope='actual_local_contract_only',task_recovery_success_claim=False,
        formal_training_ready=False)
    a.output.with_suffix('.build_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
