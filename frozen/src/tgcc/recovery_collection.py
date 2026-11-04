"""Measured geometry-recovery collection helpers, independent of policy labels.

These utilities certify only the measured position-return contract. They never
infer grasp, task completion, or a corrective policy label from a perturbation.
"""

from dataclasses import asdict, dataclass
import math
import statistics


def body_velocity_to_world_target(
    position, quaternion_xyzw, target_xy, gain=2.0, limit=0.04
):
    """SE(2) world position error -> bounded native base-frame m/s command."""
    if gain <= 0 or limit <= 0:
        raise ValueError("gain and limit must be positive")
    x, y, z, w = map(float, quaternion_xyzw)
    values = [*position, x, y, z, w, *target_xy, gain, limit]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("non-finite target or feedback")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("orientation must be a unit quaternion")
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    ex, ey = (
        float(target_xy[0]) - float(position[0]),
        float(target_xy[1]) - float(position[1]),
    )
    vx = gain * (math.cos(yaw) * ex + math.sin(yaw) * ey)
    vy = gain * (-math.sin(yaw) * ex + math.cos(yaw) * ey)
    scale = min(1.0, limit / max(math.hypot(vx, vy), 1e-12))
    return [vx * scale, vy * scale, 0.0]


def velocity_to_controller_input(velocity, contract):
    """Invert audited base command scaling: SI velocities are not raw inputs."""
    entry = contract["controllers"]["base"]
    if entry["motor_type"] != "velocity":
        raise ValueError("base controller must accept velocity targets")
    lower_in, upper_in = entry["command_input_limits"]
    lower_out, upper_out = entry["command_output_limits"]
    if any(len(x) != 3 for x in (velocity, lower_in, upper_in, lower_out, upper_out)):
        raise ValueError("three-axis audited base limits required")
    result = []
    for v, li, ui, lo, uo in zip(velocity, lower_in, upper_in, lower_out, upper_out):
        if (
            not all(math.isfinite(float(x)) for x in (v, li, ui, lo, uo))
            or ui <= li
            or uo <= lo
        ):
            raise ValueError("invalid base command scaling")
        if not lo <= v <= uo:
            raise ValueError("desired velocity exceeds actual controller limits")
        result.append(li + (v - lo) / (uo - lo) * (ui - li))
    return result


@dataclass(frozen=True)
class GeometryRecoveryCertificate:
    scope: str
    verified: bool
    reason: str
    initial_error_m: float
    final_error_m: float
    executed_steps: int
    maximum_unrelated_object_displacement_m: float
    valid_trial: bool = False


def certify_position_return(
    *,
    initial_error_m,
    final_error_m,
    executed_steps,
    maximum_unrelated_object_displacement_m,
    restore_verified,
    terminal,
    threshold_m=0.005,
):
    """Reject ineffective perturbations, collateral changes, and fake execution."""
    metrics = [
        initial_error_m,
        final_error_m,
        maximum_unrelated_object_displacement_m,
        threshold_m,
    ]
    if not all(math.isfinite(float(v)) and v >= 0 for v in metrics):
        raise ValueError("invalid measured geometry")
    reason = "verified_position_return"
    if not restore_verified:
        reason = "state_restore_unverified"
    elif terminal:
        reason = "environment_terminated"
    elif executed_steps <= 0:
        reason = "no_actual_execution"
    elif initial_error_m < 2 * threshold_m:
        reason = "perturbation_not_resolved"
    elif maximum_unrelated_object_displacement_m > 0.002:
        reason = "unrelated_objects_changed"
    elif final_error_m > threshold_m or final_error_m >= initial_error_m:
        reason = "position_return_failed"
    return GeometryRecoveryCertificate(
        "base_xy_return_only",
        reason == "verified_position_return",
        reason,
        float(initial_error_m),
        float(final_error_m),
        int(executed_steps),
        float(maximum_unrelated_object_displacement_m),
        reason in ("verified_position_return", "position_return_failed"),
    )


def timing_summary(records):
    """Report both attempted and accepted costs; no throughput extrapolation."""
    values = [float(r["wall_seconds"]) for r in records]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("invalid measured wall time")
    accepted = sum(
        bool(r["certificate"].get("valid_trial", r["certificate"]["verified"]))
        for r in records
    )
    total = sum(values)
    return {
        "attempted": len(records),
        "accepted": accepted,
        "successful_recoveries": sum(
            bool(r["certificate"]["verified"]) for r in records
        ),
        "acceptance_rate": accepted / len(records) if records else None,
        "seconds_per_attempt_median": statistics.median(values) if values else None,
        "seconds_per_accepted_amortized": total / accepted if accepted else None,
        "collection_wall_seconds": total,
        "includes_startup": False,
        "supports_task_or_grasp_recovery_claim": False,
    }


def certificate_dict(certificate):
    return asdict(certificate)


def certify_joint_return(
    initial_error_rad,
    final_error_rad,
    executed_steps,
    unrelated_motion_m,
    *,
    restore_verified,
    terminal,
):
    """Only certify measured joint-space tracking, not arbitrary Cartesian IK."""
    values = (initial_error_rad, final_error_rad, unrelated_motion_m)
    if not all(math.isfinite(float(x)) and x >= 0 for x in values):
        raise ValueError("invalid measured joint error")
    valid = (
        restore_verified
        and not terminal
        and executed_steps > 0
        and initial_error_rad >= 0.02
        and unrelated_motion_m <= 0.002
    )
    verified = valid and final_error_rad <= 0.01
    return dict(
        scope="arm_joint_return_only",
        verified=bool(verified),
        valid_trial=bool(valid),
        initial_error_rad=float(initial_error_rad),
        final_error_rad=float(final_error_rad),
        executed_steps=int(executed_steps),
        maximum_unrelated_object_displacement_m=float(unrelated_motion_m),
        reason="verified_joint_return" if verified else "joint_return_not_verified",
    )
