"""TGR-owned protocol, adapted from audited baseline codec; no baseline imports.
Lossless WebP224, atomic action+RNG+graph cache; socket retries never step physics.
"""

"""msgpack codec with numpy support, shared by serve.py and policy ws clients.

Wire format (matches OpenPI's hook so the two are interoperable):

    ndarray  → {"__ndarray__": True, "data": <bytes>, "dtype": str, "shape": tuple}
    np.generic → {"__npgeneric__": True, "data": <python scalar>, "dtype": str}

Decoder side recursively turns bytes keys into str keys, so consumers can use
the natural ``obj["key"]`` access regardless of the writer's locale.
"""


import functools
from typing import Any

import msgpack
import numpy as np


_UNSUPPORTED_KINDS = ("V", "O", "c")  # void, object, complex


def _pack(obj: Any) -> Any:
    if (
        isinstance(obj, (np.ndarray, np.generic))
        and obj.dtype.kind in _UNSUPPORTED_KINDS
    ):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {"__npgeneric__": True, "data": obj.item(), "dtype": obj.dtype.str}
    return obj


def _bytes_to_str_keys(obj: Any) -> Any:
    """Recursively decode bytes-typed dict keys to str."""
    if isinstance(obj, dict):
        return {
            (k.decode() if isinstance(k, bytes) else k): _bytes_to_str_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_bytes_to_str_keys(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_bytes_to_str_keys(v) for v in obj)
    return obj


def _unpack(obj: dict[Any, Any]) -> Any:
    obj = _bytes_to_str_keys(obj)
    if "__ndarray__" in obj:
        return np.ndarray(
            buffer=obj["data"],
            dtype=np.dtype(obj["dtype"]),
            shape=tuple(obj["shape"]),
        )
    if "__npgeneric__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack)

import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(
    images: np.ndarray, height: int, width: int, method=Image.BILINEAR
) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            _resize_with_pad_pil(Image.fromarray(im), height, width, method=method)
            for im in images
        ]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(
    image: Image.Image, height: int, width: int, method: int
) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


"""Move the exact existing model224 resize before upload; preserve raw recordings."""
import numpy as np


def for_network(raw):
    if not raw.get("images"):
        return raw
    result = dict(raw)
    result["images"] = {}
    for key, value in raw["images"].items():
        im = np.asarray(value)
        if im.dtype != np.uint8 or im.ndim != 3 or im.shape[0] != 3:
            raise ValueError("Expected captured CHW uint8 RGB")
        result["images"][key] = np.ascontiguousarray(
            resize_with_pad(im.transpose(1, 2, 0), 224, 224).transpose(2, 0, 1)
        )
    return result


"""Wire-compatible v3: valid model replies proceed without extra echo traffic."""
import io, time, zlib
from collections import deque
import numpy as np
from PIL import Image


MAGIC = b"TGRW1\0"
OLD_MAGIC = b"PI05Z1\0"


def encode(raw):
    net = for_network(raw)
    if net.get("images"):
        net = dict(net)
        images = {}
        for k, a in net["images"].items():
            f = io.BytesIO()
            Image.fromarray(a.transpose(1, 2, 0)).save(
                f, format="WEBP", lossless=True, method=3, exact=True
            )
            images[k] = f.getvalue()
        net["images"] = images
    return MAGIC + packb(net)


def decode(data):
    if data.startswith(MAGIC):
        raw = unpackb(data[len(MAGIC) :])
        if raw.get("images"):
            raw["images"] = {
                k: np.ascontiguousarray(
                    np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).transpose(
                        2, 0, 1
                    )
                )
                for k, b in raw["images"].items()
            }
            if any(a.shape != (3, 224, 224) for a in raw["images"].values()):
                raise ValueError("Expected lossless224 images")
        return raw
    return unpackb(
        zlib.decompress(data[len(OLD_MAGIC) :]) if data.startswith(OLD_MAGIC) else data
    )


def split_action(row):
    a = np.asarray(row, dtype=np.float32)
    if a.shape != (23,) or not np.isfinite(a).all():
        raise ValueError("Invalid 23D action")
    return dict(
        lower_body=a[:7],
        left_arm=a[7:14],
        left_gripper=a[14:15],
        right_arm=a[15:22],
        right_gripper=a[22:23],
    )


"""Durable exactly-once policy chunks; no simulator, task logic or smoothing."""
import hashlib, os, pathlib
from collections import OrderedDict
import numpy as np


class SessionStore:
    def __init__(
        self,
        path,
        weight,
        seed_rng,
        get_rng,
        set_rng,
        get_aux=lambda: None,
        set_aux=lambda x: None,
    ):
        self.get_aux = get_aux
        self.set_aux = set_aux
        self.path = pathlib.Path(path)
        self.weight = weight
        self.seed_rng = seed_rng
        self.get_rng = get_rng
        self.set_rng = set_rng
        self.state = None
        if self.path.exists():
            state = unpackb(self.path.read_bytes())
            assert state["weight"] == weight, "Durable state weight mismatch"
            self.state = state
            self.set_rng(state["rng"])
            self.set_aux(state.get("aux"))

    def commit(self):
        self.state["rng"] = self.get_rng()
        self.state["aux"] = self.get_aux()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        t = self.path.with_suffix(".tmp")
        with t.open("wb") as f:
            f.write(packb(self.state))
            f.flush()
            os.fsync(f.fileno())
        os.replace(t, self.path)
        fd = os.open(self.path.parent, os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)

    def start(self, session, seed):
        if self.state and self.state["session"] == session:
            if self.state["seed"] != seed:
                raise ValueError("Session seed changed")
            return self.ack()
        self.seed_rng(seed)
        self.state = {
            "weight": self.weight,
            "session": session,
            "seed": seed,
            "next": 0,
            "cache": [],
        }
        self.commit()
        return self.ack()

    def ack(self):
        return {
            "session": self.state["session"],
            "next": self.state["next"],
            "seed": self.state["seed"],
            "resumed": True,
        }

    def resume(self, session):
        if not self.state or self.state["session"] != session:
            raise RuntimeError(
                "SESSION_MISSING: refuse to reset an active physical task"
            )
        return self.ack()

    def infer(self, session, seq, digest, run):
        self.resume(session)
        for item in self.state["cache"]:
            if item["id"] == seq:
                if item["digest"] != digest:
                    raise ValueError("Duplicate request changed payload")
                return dict(item["reply"], cached=True)
        if seq != self.state["next"]:
            raise ValueError("Unexpected request sequence")
        # Detach rollback snapshots: runtime exports can hold mutable arrays/lists.
        before_rng = unpackb(packb(self.get_rng()))
        before_aux = unpackb(packb(self.get_aux()))
        before_state = unpackb(packb(self.state))
        try:
            actions, seconds = run()
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != (32, 23) or not np.isfinite(actions).all():
                raise ValueError("Invalid 32x23 model action chunk")
            reply = {
                "session": session,
                "id": seq,
                "actions": actions,
                "inference_seconds": seconds,
                "cached": False,
            }
            self.state["cache"].append({"id": seq, "digest": digest, "reply": reply})
            self.state["cache"] = self.state["cache"][-4:]
            self.state["next"] = seq + 1
            self.commit()  # Persist before replying: lost replies replay without advancing RNG.
            return reply
        except BaseException:
            # A post-rename fsync failure must recover the durable reply, not rerun it.
            durable = (
                unpackb(self.path.read_bytes()) if self.path.exists() else before_state
            )
            if durable.get("session") == session and durable.get("next") == seq + 1:
                self.state = durable
                self.set_rng(durable["rng"])
                self.set_aux(durable.get("aux"))
            else:
                self.state = before_state
                self.set_rng(before_rng)
                self.set_aux(before_aux)
            raise


"""One active rollout: same-scene reconnect + byte-identical chunk request retry."""
import hashlib, socket, struct, time, uuid
from collections import deque
import numpy as np


class ReliableClient:
    def __init__(
        self,
        connect,
        weight,
        seed=42,
        session=None,
        on_event=None,
        recovery_seconds=600,
    ):
        self.connect = connect
        self.weight = weight
        self.seed = seed
        self.session = session or uuid.uuid4().hex
        self.event = on_event or (lambda e: None)
        self.recovery_seconds = recovery_seconds
        self.ws = None
        self.started = False
        self.pending = deque()
        self.seq = 0
        self.last_timing = None

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def open(self):
        self.ws = self.connect()
        meta = decode(self.ws.recv(timeout=20))
        if (
            meta.get("weight_sha256") != self.weight
            or meta.get("transport") != "tgr_durable_v1"
        ):
            raise ValueError("Refuse different checkpoint/transport")
        if hasattr(self.ws, "socket"):
            try:
                self.ws.socket.setsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDTIMEO, struct.pack("ll", 20, 0)
                )
            except (OSError, AttributeError):
                pass
        op = {
            "op": "resume" if self.started else "start",
            "session": self.session,
            "seed": self.seed,
        }
        self.ws.send(packb(op))
        ack = decode(self.ws.recv(timeout=20))
        if "error" in ack:
            raise ValueError(ack["error"])
        if ack.get("session") != self.session:
            raise ValueError("Session acknowledgment mismatch")
        self.started = True

    def request(self, raw):
        if raw:
            if self.pending:
                raise ValueError("Unexecuted actions must not be discarded")
            payload = dict(raw, op="infer", session=self.session, id=self.seq)
            t = time.monotonic()
            message = encode(payload)
            encode_seconds = time.monotonic() - t
            digest = hashlib.sha256(message).hexdigest()
            deadline = t + self.recovery_seconds
            attempt = 0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Infrastructure: bounded same-scene reconnect exhausted"
                    )
                try:
                    if self.ws is None:
                        self.open()
                    self.ws.send(message)
                    reply = decode(
                        self.ws.recv(
                            timeout=min(35, max(1, deadline - time.monotonic()))
                        )
                    )
                    if "error" in reply:
                        raise ValueError(reply["error"])
                    if (
                        reply.get("session") != self.session
                        or reply.get("id") != self.seq
                    ):
                        raise ValueError("Response request identity mismatch")
                    a = np.asarray(reply["actions"], dtype=np.float32)
                    if a.shape != (32, 23) or not np.isfinite(a).all():
                        raise ValueError("Invalid model chunk")
                    self.pending.extend(a.copy())
                    elapsed = time.monotonic() - t
                    self.last_timing = {
                        "id": self.seq,
                        "query_seconds": elapsed,
                        "inference_seconds": reply["inference_seconds"],
                        "upload_bytes": len(message),
                        "encode_seconds": encode_seconds,
                        "cached": bool(reply.get("cached")),
                        "recoveries": attempt,
                        "request_sha256": digest,
                    }
                    self.seq += 1
                    if attempt:
                        self.event(
                            {
                                "event": "recovered",
                                "same_scene": True,
                                "no_policy_reset": True,
                                "cached": reply.get("cached"),
                                "id": self.seq - 1,
                                "attempts": attempt,
                                "seconds": elapsed,
                            }
                        )
                    break
                except ValueError:
                    raise  # A model/schema/session error is not a network error.
                except Exception as exc:
                    attempt += 1
                    self.event(
                        {
                            "event": "reconnecting",
                            "id": self.seq,
                            "attempt": attempt,
                            "error": repr(exc),
                            "same_scene": True,
                            "no_step": True,
                            "request_sha256": digest,
                        }
                    )
                    self.close()
                    time.sleep(min(8, attempt * 2))
        elif not self.pending:
            raise ValueError("Fresh observation required")
        return {
            "action": split_action(self.pending.popleft()),
            "need_obs": not self.pending,
        }
