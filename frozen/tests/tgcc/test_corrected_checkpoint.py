import json
import pytest
import torch.multiprocessing as mp
import test_distribution_checkpoint as fixture


def worker(rank,world,init_file,root,step):
    original=fixture.metadata
    def metadata(world):
        md=original(world)
        md['config'].update(schema='tgcc.corrected_full_success_distribution.recipe.v1',
                            critic_initialization_actual_updates=0,checkpoint_storage='native_zstd',
                            new_updates=5000,joint_updates=4000)
        md.update(policy_initialization_updates=10000,feature_schema='corrected-delta',
                  execution_target='actual_after_minus_actual_before',memory_unroll_blocks=8,
                  memory_burnin_blocks=4,audit_phase=None,student_initialization_sha256='a'*64,
                  feature_library_sha256='b'*64)
        return md
    fixture.metadata=metadata
    fixture.worker(rank,world,init_file,root,step,True)


@pytest.mark.parametrize('step',[1,1002],ids=['fresh-critic-warmup','full-joint-after-warmup'])
def test_corrected_fresh_critic_and_inherited_policy_counters_restore_exactly(tmp_path,step):
    mp.spawn(worker,args=(2,str(tmp_path/'gloo_init'),str(tmp_path),step),nprocs=2,join=True)
    assert all(all(json.loads((tmp_path/f'rank{rank}.json').read_text()).values()) for rank in range(2))
    md=json.loads((tmp_path/'checkpoints/latest/metadata.json').read_text())
    assert md['full_model_updates']==max(0,step-1000)
    assert md['fresh_critic_only_updates']==min(step,1000)
    assert md['critic_actual_total_updates']==step
    assert md['policy_cumulative_updates']==10000+max(0,step-1000)
    assert md['memory_unroll_blocks']==8
