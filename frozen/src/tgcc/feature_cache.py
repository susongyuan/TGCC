"""Bounded-disk memoization of deterministic frozen-parent feature pairs only."""

from pathlib import Path
from dataclasses import fields
import os, shutil, uuid
import torch
from .specs import CriticObservation


def slice_observation(o, i, device="cpu", compact=False):
    def select(v, name):
        if v is None:
            return None
        if isinstance(v, tuple):
            return tuple(select(x, name) for x in v)
        value = v[i : i + 1].detach().to(device)
        # Quantization is applied identically on fresh and cached critic routes.
        if compact and name in ("graph_tokens", "semantic_context"):
            value = value.half()
        return value

    return CriticObservation(
        **{f.name: select(getattr(o, f.name), f.name) for f in fields(o)}
    )


class FeatureCache:
    def __init__(self, path, identity):
        self.root = Path(path) / identity[:24]
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, index):
        return self.root / f"{index // 1000:03d}" / f"{index:06d}.pt"

    def load(self, index):
        p = self.path(index)
        return (
            torch.load(p, map_location="cpu", weights_only=False)
            if p.exists()
            else None
        )

    def save(self, index, before, after, native):
        p = self.path(index)
        if p.exists():
            return
        if shutil.disk_usage(self.root).free < 51 * 2**30:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix("." + uuid.uuid4().hex + ".tmp")
        torch.save(dict(before=before, after=after, native=native.detach().cpu()), tmp)
        os.replace(tmp, p)


class CachedDemonstrations:
    def __init__(self, metadata, cache, use_cache=True):
        self.metadata = metadata
        self.cache = cache
        self.use_cache = use_cache

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        cached = self.cache.load(index) if self.use_cache else None
        return dict(
            index=index, cached=cached, sample=None if cached else self.metadata[index]
        )
