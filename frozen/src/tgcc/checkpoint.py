"""Atomic, exact continuation for TGCC-only state with immutable parent identity."""

from pathlib import Path
import hashlib, json, os, random, shutil
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file, load_file


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def sources(root):
    root = Path(root)
    folders = (
        "src",
        "tools/tgcc",
        "configs",
        "deployment",
        "scripts",
        "code/src",
        "code/packages",
        "reports",
        "data/robot",
        "data/task_definitions",
    )
    # Training dataset descriptors are recorded independently; an inference
    # install must not need private demonstration manifests or RGB files.
    return {
        str(p.relative_to(root)): digest(p)
        for folder in folders
        for p in sorted((root / folder).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".json", ".urdf", ".bddl")
    }


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state(s["cuda"])


def save_checkpoint(root, model, optimizer, step, cursor, metadata, rank=0):
    root = Path(root)
    tmp = root / f".step{step:05d}.building"
    final = root / f"step{step:05d}"
    distributed = dist.is_initialized()
    if rank == 0:
        if final.exists():
            raise FileExistsError(final)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
    if distributed:
        dist.barrier()
    torch.save(rng_state(), tmp / f"rng_rank{rank}.pt")
    if rank == 0:
        weights = {
            k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
        }
        save_file(weights, str(tmp / "tgcc.safetensors"))
        torch.save(
            dict(
                optimizer=optimizer.state_dict(), new_updates=step, data_cursor=cursor
            ),
            tmp / "resume.pt",
        )
        md = dict(
            metadata,
            schema="tgcc.checkpoint.v1",
            new_updates=step,
            optimizer_steps=5000 + step,
            effective_sequence_exposures=step * 48,
            equivalent_passes=step * 48 / 51714,
            checkpoint_sha256=digest(tmp / "tgcc.safetensors"),
            optimizer_reset_reason="new critic and adapters; parent optimizer snapshot is not selected step5000",
            ema=None,
            training_complete=step == 5000,
        )
        (tmp / "metadata.json").write_text(json.dumps(md, indent=2))
    if distributed:
        dist.barrier()
    if rank == 0:
        os.replace(tmp, final)
        link = root / ".latest.new"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(final.name, target_is_directory=True)
        os.replace(link, root / "latest")
    if distributed:
        dist.barrier()
    return final


def load_checkpoint(path, model, optimizer, expected_sources, rank=0):
    path = Path(path)
    md = json.loads((path / "metadata.json").read_text())
    if (
        md["source_sha256"] != expected_sources
        or digest(path / "tgcc.safetensors") != md["checkpoint_sha256"]
    ):
        raise ValueError("Resume source/checkpoint identity changed")
    model.load_state_dict(load_file(str(path / "tgcc.safetensors")), strict=True)
    state = torch.load(
        path / "resume.pt",
        map_location=next(model.parameters()).device,
        weights_only=False,
    )
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(
        torch.load(path / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False)
    )
    if state["new_updates"] != md["new_updates"]:
        raise ValueError("Resume counter mismatch")
    return state["new_updates"], state["data_cursor"]
