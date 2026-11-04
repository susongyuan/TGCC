"""Success-only episode-grouped conformal thresholds and artifact readiness.

Calibration controls false alarms on held-out successful demonstrations only.
It supplies no failed-action recall, task success probability or causal reason.
"""

import hashlib
import math
from pathlib import Path



def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def calibrate_success_scores(records, index_episodes, *, alpha=0.1,
                             minimum_episodes=16, required_heads=("action", "execution", "planning")):
    if not 0 < alpha < 1:
        raise ValueError("Calibration alpha must lie in (0,1)")
    index_episodes = {str(e) for e in index_episodes}
    groups = {h: {} for h in required_heads}
    seen = set()
    execution_prefix_counts = set()
    for row in records:
        ep = str(row["episode_id"])
        if ep in index_episodes or row["split"] != "validation":
            raise ValueError("Calibration cannot use indexed/training/test episodes")
        key = (ep, row["sequence_steps"][0], row["sequence_steps"][1])
        if key in seen:
            raise ValueError("Repeated calibration sequence is not independent evidence")
        seen.add(key)
        for head in required_heads:
            if not row.get(head + "_valid", True):
                continue
            if head == "execution":
                count = row.get("executed_steps")
                if type(count) is not int or count != 32:
                    raise ValueError("Execution calibration requires the actual 32-step held-out prefix")
                execution_prefix_counts.add(count)
            score = float(row[head + "_nll"])
            if not math.isfinite(score):
                raise ValueError("Nonfinite valid calibration likelihood")
            groups[head].setdefault(ep, []).append(score)
    heads = {}
    for head, episodes in groups.items():
        maxima = sorted(max(values) for values in episodes.values())
        n = len(maxima)
        rank = math.ceil((n + 1) * (1 - alpha))
        enough = n >= minimum_episodes and rank <= n
        heads[head] = dict(enabled=enough,
            threshold=maxima[rank - 1] if enough else None,
            alpha=alpha, grouping="episode_max", calibration_episode_count=n,
            calibration_sample_count=sum(map(len, episodes.values())),
            finite_sample_quantile_rank=rank,
            disabled_reason=None if enough else "insufficient_independent_success_episodes_for_finite_threshold",
            detection_scope="success_demonstration_distribution_deviation",
            failed_example_count=0, failure_recall=None, task_success_probability=False)
        if head == "execution":
            heads[head]["calibrated_prefix_counts"] = sorted(execution_prefix_counts)
    return dict(schema="tgcc.distribution_calibration.v1", heads=heads,
        calibration_episodes=sorted({str(r["episode_id"]) for r in records}),
        index_episodes=sorted(index_episodes),
        evaluation_instances_used=False, failure_labels_used=False,
        iid_caveat="Episode quantile requires calibration/deployment episodes to be exchangeable; environment shifts void that guarantee",
        interpretation="High conditional NLL is OOD, not proof of action or subtask failure")


def distribution_readiness(metadata, feature_metadata, calibration=None):
    """Return explicit blocking errors; do not infer missing source identities."""
    errors = []
    if feature_metadata.get('context_width')==3097:
        from .transition_features import FEATURE_SCHEMA,EXECUTION_TARGET
        if (feature_metadata.get('feature_schema')!=FEATURE_SCHEMA or
                metadata.get('feature_schema')!=FEATURE_SCHEMA or
                feature_metadata.get('execution_target')!=EXECUTION_TARGET or
                metadata.get('execution_target')!=EXECUTION_TARGET):
            errors.append('corrected causal feature/target schema mismatch')
    if metadata.get("schema") != "tgcc.distribution_joint.v1":
        errors.append("wrong checkpoint recipe schema")
    for key in ("feature_codec_sha256", "retrieval_index_sha256", "demonstrations_manifest_sha256",
                "normalization_sha256", "encoder_model_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or len(value) != 64 or value != feature_metadata.get(key):
            errors.append("feature/checkpoint identity mismatch: " + key)
    indexed = {str(e) for e in feature_metadata.get("index_episodes", [])}
    validation = {str(e) for e in feature_metadata.get("validation_episodes", [])}
    if not indexed or not validation or indexed & validation:
        errors.append("missing or overlapping train/validation source episodes")
    if feature_metadata.get("retrieval_excludes_query_episode") is not True:
        errors.append("retrieval does not exclude query's whole source episode")
    if feature_metadata.get("encoder_frozen") is not True:
        errors.append("retrieval encoder is not frozen")
    if feature_metadata.get("failed_or_synthetic_targets_used", True):
        errors.append("failed/synthetic targets admitted to success-only training")
    for key in ("action", "execution", "planning"):
        coverage = feature_metadata.get("valid_training_targets", {}).get(key, 0)
        if coverage <= 0:
            errors.append("missing successful demonstration target: " + key)
    if feature_metadata.get("actual_prefix_time_alignment_checked") is not True:
        errors.append("actual execution/after-exposure alignment not verified")
    if feature_metadata.get("future_targets_as_context_inputs", True):
        errors.append("future target/context isolation not verified")
    if calibration is not None:
        if feature_metadata.get('context_width')==3097:
            for key in ('feature_schema','execution_target'):
                if calibration.get(key)!=metadata.get(key):errors.append('calibration feature schema mismatch: '+key)
        if calibration.get("schema") != "tgcc.distribution_calibration.v1":
            errors.append("wrong likelihood calibration schema")
        for key in ("feature_codec_sha256", "retrieval_index_sha256", "demonstrations_manifest_sha256"):
            if calibration.get(key) != metadata.get(key):
                errors.append("calibration/checkpoint identity mismatch: " + key)
        if not set(calibration.get("calibration_episodes", [])).issubset(validation):
            errors.append("calibration sources are not held-out validation episodes")
        for name, head in calibration.get("heads", {}).items():
            enabled = head.get("enabled", False)
            threshold = head.get("threshold")
            if type(enabled) is not bool or (enabled and (
                    type(threshold) not in (float, int) or not math.isfinite(threshold))):
                errors.append("invalid calibration head enable/finite threshold: " + name)
            if name == "execution" and enabled and head.get("calibrated_prefix_counts") != [32]:
                errors.append("execution calibration does not bind the actual 32-step prefix")
        if {str(e) for e in calibration.get("calibration_episodes", [])} & indexed:
            errors.append("calibration episode leakage into index")
    return dict(status="failed" if errors else "passed", failures=errors,
        success_only=True, normality_is_not_task_success=True)


def bind_calibration(result, metadata, distribution_weights_path):
    bound = dict(result)
    for key in ("feature_codec_sha256", "retrieval_index_sha256", "demonstrations_manifest_sha256",
                "normalization_sha256", "encoder_model_sha256", "source_sha256"):
        bound[key] = metadata[key]
    bound["distribution_sha256"] = digest_file(distribution_weights_path)
    for key in ('feature_schema','execution_target','memory_unroll_blocks'):
        if key in metadata:bound[key]=metadata[key]
    return bound
