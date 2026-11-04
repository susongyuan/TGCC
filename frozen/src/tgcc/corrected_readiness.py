"""Correction-specific admission: source reviews cannot replace GPU evidence."""


def validate_corrected_readiness(gate,metadata):
    bound=('source_sha256','config','feature_library_sha256','world_size','micro_batch',
           'student_initialization_sha256','memory_unroll_blocks','memory_burnin_blocks')
    if gate.get('status')!='passed' or any(gate.get(k)!=metadata[k] for k in bound):
        raise ValueError('Corrected readiness identity missing or stale')
    required=('full_data_gate_passed','real_unrolled_data_gate_passed',
              'reference_condition_validation_passed','actual_next_complete_update_exact')
    if any(gate.get(k) is not True for k in required) or gate.get('clean_source_review_passes')!=3:
        raise ValueError('Corrected data/reference/three-clean-review gates incomplete')
    for phase in ('warmup','joint'):
        records=gate.get('gpu_phase_audits',{}).get(phase,[])
        world=metadata['world_size']
        if world not in (3,4) or len(records)!=world or {r.get('rank') for r in records}!=set(range(world)):
            raise ValueError('Both complete GPU-world audit phases are required')
        for record in records:
            if (record.get('status')!='passed' or record.get('audit_phase')!=phase
                    or any(record.get(k)!=metadata[k] for k in bound)
                    or record.get('actual_next_complete_update_exact') is not True
                    or record.get('actual_model_adam_rng_roundtrip') is not True
                    or record.get('formal_optimizer_updates')!=0):
                raise ValueError('Unbound/incomplete GPU phase audit')
