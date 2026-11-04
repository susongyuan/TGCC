"""Executed-command receipts are acknowledged AFTER successful physics steps."""

import hashlib
import time

import numpy as np
from tgr.transport import ReliableClient, SessionStore, decode, encode, packb, unpackb


class ControllerCapabilityStop(ValueError):
    """A bounded controller stop has no natural task success/Q label."""

    def __init__(self, reply):
        self.reply = reply
        super().__init__(reply["error"])


class _StatusSocket:
    """Preserve structured server stop status before the legacy codec drops it."""

    def __init__(self, socket):
        self.wrapped = socket

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def recv(self, *args, **kwargs):
        message = self.wrapped.recv(*args, **kwargs)
        reply = decode(message)
        if (isinstance(reply, dict) and reply.get("status") == "capability_stop"
                and reply.get("transaction_rolled_back") is True
                and reply.get("task_failure", False) is None and reply.get("Q", False) is None
                and isinstance(reply.get("error"), str)):
            raise ControllerCapabilityStop(reply)
        return message


class ExecutedClient(ReliableClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        connect = self.connect
        self.connect = lambda: _StatusSocket(connect())
        self.executed = []
        self.exposure_time = None
        self.last_native = None
        self.finalized = False

    def request(self, raw):
        if self.finalized:
            raise ValueError("Finalized execution cannot request another action")
        if raw:
            raw = dict(raw)
            raw["native_hold"] = native_hold(
                raw, self.executed[-1] if self.executed else None
            )
            if self.exposure_time is not None:
                if not self.executed:
                    raise ValueError("New exposure without acknowledged execution")
                raw["execution_receipt"] = dict(
                    start_time=self.exposure_time,
                    end_time=float(raw["sim_time"]),
                    actions=np.stack(self.executed).astype(np.float32),
                    step_count=len(self.executed),
                    previous_request_id=self.seq - 1,
                    source="native_controller_commands_after_successful_env_step",
                )
            result = self._request_with_status(raw)
            self.executed = []
            self.exposure_time = float(raw["sim_time"])
        else:
            result = self._request_with_status(raw)
        p = result["action"]
        self.last_native = np.r_[
            p["lower_body"],
            p["left_arm"],
            p["left_gripper"],
            p["right_arm"],
            p["right_gripper"],
        ].astype(np.float32)
        return result

    def finalize(self, raw):
        """Send the actual final after-state; retain the unexecuted suffix.

        Lost replies retry byte-identically and cannot generate or execute any
        action. Independent simulator metrics remain the task-result authority.
        """
        if self.finalized or self.last_native is not None or not self.executed:
            raise ValueError("Final assessment needs acknowledged final execution")
        raw = dict(raw, op="finalize", session=self.session, id=self.seq)
        raw["execution_receipt"] = dict(
            start_time=self.exposure_time, end_time=float(raw["sim_time"]),
            actions=np.stack(self.executed).astype(np.float32), step_count=len(self.executed),
            previous_request_id=self.seq - 1,
            source="native_controller_commands_after_successful_env_step")
        message = encode(raw)
        digest = hashlib.sha256(message).hexdigest()
        deadline = time.monotonic() + self.recovery_seconds
        attempt = 0
        while time.monotonic() < deadline:
            try:
                if self.ws is None:
                    self.open()
                self.ws.send(message)
                reply = decode(self.ws.recv(timeout=min(35, max(1, deadline - time.monotonic()))))
                if "error" in reply:
                    raise ValueError(reply["error"])
                if (reply.get("session") != self.session or reply.get("id") != self.seq
                        or reply.get("finalized") is not True or "actions" in reply
                        or reply.get("new_action_generated") is not False):
                    raise ValueError("Final execution assessment identity/schema mismatch")
                self.finalized = True
                self.seq += 1
                self.executed = []
                return reply
            except ValueError:
                raise
            except Exception as exc:
                attempt += 1
                self.event(dict(event="final_assessment_reconnecting", id=self.seq,
                                attempt=attempt, error=repr(exc), same_scene=True,
                                no_step=True, request_sha256=digest))
                self.close()
                time.sleep(min(8, attempt * 2, max(0, deadline - time.monotonic())))
        raise TimeoutError("Infrastructure: bounded final-assessment reconnect exhausted")

    def _request_with_status(self, raw):
        return super().request(raw)

    def acknowledge_step(self, actual_action, controller_indices):
        if self.last_native is None:
            raise ValueError("No dispatched action to acknowledge")
        a = np.asarray(actual_action, dtype=np.float32)
        native = np.r_[
            a[controller_indices["base"]],
            a[controller_indices["trunk"]],
            a[controller_indices["arm_left"]],
            a[controller_indices["gripper_left"]],
            a[controller_indices["arm_right"]],
            a[controller_indices["gripper_right"]],
        ]
        if native.shape != (23,) or not np.array_equal(native, self.last_native):
            raise ValueError(
                "Executed controller permutation differs from dispatched native action"
            )
        self.executed.append(native.copy())
        self.last_native = None


class ExecutedSessionStore(SessionStore):
    """Exactly-once final assessment using the existing durable transaction."""

    def infer(self, session, seq, digest, run):
        if self.state and self.state.get("finalized"):
            raise ValueError("Finalized execution cannot generate another action")
        return super().infer(session, seq, digest, run)

    def finalize(self, session, seq, digest, run):
        self.resume(session)
        for item in self.state["cache"]:
            if item["id"] == seq:
                if item["digest"] != digest or item.get("op") != "finalize":
                    raise ValueError("Duplicate final assessment changed payload/operation")
                return dict(item["reply"], cached=True)
        if self.state.get("finalized") or seq != self.state["next"]:
            raise ValueError("Unexpected final assessment sequence")
        before_rng, before_aux, before_state = (unpackb(packb(value)) for value in
                                                 (self.get_rng(), self.get_aux(), self.state))
        try:
            assessment, seconds = run()
            if (assessment.get("final_execution_assessment") is not True
                    or assessment.get("new_action_generated") is not False):
                raise ValueError("Final assessment cannot generate another action")
            reply = dict(session=session, id=seq, finalized=True, new_action_generated=False,
                         final_execution_assessment=assessment, inference_seconds=seconds, cached=False)
            self.state["cache"].append(dict(id=seq, digest=digest, reply=reply, op="finalize"))
            self.state["cache"] = self.state["cache"][-4:]
            self.state.update(next=seq + 1, finalized=True)
            self.commit()
            return reply
        except BaseException:
            durable = unpackb(self.path.read_bytes()) if self.path.exists() else before_state
            if durable.get("session") == session and durable.get("next") == seq + 1:
                self.state = durable
                self.set_rng(durable["rng"])
                self.set_aux(durable.get("aux"))
            else:
                self.state = before_state
                self.set_rng(before_rng)
                self.set_aux(before_aux)
            raise


def native_hold(raw, previous=None):
    """Invert actual smooth-gripper scaling; retain grasp commands when held."""
    s = raw["state"]
    c = raw["control_contract"]["controllers"]
    result = np.r_[
        np.zeros(3), s["trunk_qpos"], s["left_arm"], 0.0, s["right_arm"], 0.0
    ].astype(np.float32)
    snapshot = raw["graph_snapshot"]
    nodes = {n["entity_id"]: n for n in snapshot["entities"]}
    for side, dim in (("left", 14), ("right", 22)):
        limits = c["gripper_" + side]
        a, b = limits["command_input_limits"]
        lo, hi = limits["command_output_limits"]
        q = np.asarray(s[side + "_gripper"]).mean()
        low = np.asarray(lo).mean()
        high = np.asarray(hi).mean()
        if high <= low:
            raise ValueError("Unknown gripper scaling")
        result[dim] = (q - low) / (high - low) * (
            np.asarray(b).mean() - np.asarray(a).mean()
        ) + np.asarray(a).mean()
        held = any(
            e["relation_type"] == "held_by"
            and e.get("relative_attributes", {}).get("value") is True
            and nodes.get(e["target_id"], {}).get("instance_id") == "robot:ee:" + side
            for e in snapshot["edges"]
        )
        if held and previous is not None:
            result[dim] = previous[dim]
    return result
