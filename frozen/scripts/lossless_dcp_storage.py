"""Lossless DCP storage using PyTorch 2.7's native stream.zstd/1 extension.

This changes only storage bytes. Model, AdamW precision, RNG and sampler are
preserved. Native FileSystemReader reconstructs the extension from metadata.
"""
import argparse
import dataclasses
import hashlib
import io
import json
import math
import os
import pickle
import shutil
import time
from pathlib import Path

GIB = 1024**3
CHUNK = 8 * 1024**2


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def writer(path):
    """Native reader requires no custom extension registry."""
    from torch.distributed.checkpoint import FileSystemWriter
    from torch.distributed.checkpoint._extension import ZStandard
    if not ZStandard.is_available():
        raise RuntimeError("native ZStandard DCP extension unavailable")
    return FileSystemWriter(path, _extensions=[ZStandard()])


def floor_bytes(path, floor_gib=50):
    free = shutil.disk_usage(path).free
    floor = math.ceil(max(50, floor_gib) * GIB)
    if free < floor:
        raise OSError(f"free floor violated: {free} < {floor}")
    return free


class GuardedOutput(io.RawIOBase):
    """Check the actual free floor before every compressed output write."""
    def __init__(self, stream, root, floor_gib):
        self.stream, self.root = stream, Path(root)
        self.floor = math.ceil(max(50, floor_gib) * GIB)

    def writable(self):
        return True

    def write(self, data):
        if shutil.disk_usage(self.root).free - len(data) < self.floor:
            raise OSError("compression stopped before consuming reserved disk floor")
        return self.stream.write(data)

    def flush(self):
        self.stream.flush()


def verify_pair(original, compressed):
    """Compare every reconstructed serialized item with original bytes by SHA."""
    import zstandard as zstd
    original, compressed = Path(original), Path(compressed)
    old = pickle.loads((original / "distributed/.metadata").read_bytes())
    new = pickle.loads((compressed / "distributed/.metadata").read_bytes())
    if old.state_dict_metadata != new.state_dict_metadata:
        raise ValueError("DCP state metadata changed")
    if old.planner_data != new.planner_data:
        raise ValueError("DCP planner metadata changed")
    if set(old.storage_data) != set(new.storage_data):
        raise ValueError("DCP storage index set changed")
    rows = []
    for index, span in old.storage_data.items():
        new_span = new.storage_data[index]
        if new_span.transform_descriptors != ["stream.zstd/1"]:
            raise ValueError("non-native compression descriptor")
        expected = hashlib.sha256()
        actual = hashlib.sha256()
        with (original / "distributed" / span.relative_path).open("rb") as f:
            f.seek(span.offset)
            left = span.length
            while left:
                part = f.read(min(CHUNK, left))
                if not part:
                    raise ValueError("original span truncated")
                expected.update(part)
                left -= len(part)
        from torch.distributed.checkpoint.filesystem import _create_file_view
        count = 0
        with (compressed / "distributed" / new_span.relative_path).open("rb") as f:
            view = _create_file_view(f, new_span.offset, new_span.length)
            with zstd.ZstdDecompressor().stream_reader(view, closefd=False) as reader:
                while part := reader.read(CHUNK):
                    actual.update(part)
                    count += len(part)
        if count != span.length or actual.digest() != expected.digest():
            raise ValueError("serialized bytes changed: " + index.fqn)
        rows.append({"fqn": index.fqn, "offset": list(index.offset) if index.offset is not None else None,
                     "original_bytes": span.length, "compressed_bytes": new_span.length,
                     "serialized_sha256": expected.hexdigest()})
    extra = {}
    for path in original.iterdir():
        if path.is_file():
            target = compressed / path.name
            if not target.is_file() or path.read_bytes() != target.read_bytes():
                raise ValueError("metadata/RNG changed: " + path.name)
            extra[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(passed=True, spans=len(rows),
                uncompressed_serialized_bytes=sum(r["original_bytes"] for r in rows),
                compressed_serialized_bytes=sum(r["compressed_bytes"] for r in rows),
                unchanged_sidecars=extra, items=rows)


def convert_resume(original, temp, *, floor_gib=50):
    """Leave the original directory intact until native-format verification passes."""
    import zstandard as zstd
    original, temp = Path(original), Path(temp)
    if temp.exists():
        raise ValueError("compression temp already exists; diagnose explicitly")
    if not (original / "metadata.json").is_file():
        raise ValueError("complete original training metadata required")
    old_blob = (original / "distributed/.metadata").read_bytes()
    old = pickle.loads(old_blob)
    if any(getattr(s, "transform_descriptors", None) for s in old.storage_data.values()):
        raise ValueError("conversion accepts only original uncompressed DCP")
    floor_bytes(original, floor_gib)
    temp.mkdir()
    dest = temp / "distributed"
    dest.mkdir()
    begin = time.time()
    new_info = {}
    grouped = {}
    for index, span in old.storage_data.items():
        grouped.setdefault(span.relative_path, []).append((index, span))
    counter = 0
    for relative, items in grouped.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or rel.name != relative:
            raise ValueError("unsafe DCP relative path")
        with (original / "distributed" / relative).open("rb") as src, (dest / relative).open("wb") as output:
            for index, span in sorted(items, key=lambda p: p[1].offset):
                src.seek(span.offset)
                start = output.tell()
                guard = GuardedOutput(output, original, floor_gib)
                with zstd.ZstdCompressor().stream_writer(guard, closefd=False) as compressor:
                    remaining = span.length
                    while remaining:
                        part = src.read(min(CHUNK, remaining))
                        if not part:
                            raise ValueError("source DCP span truncated")
                        compressor.write(part)
                        remaining -= len(part)
                length = output.tell() - start
                new_info[index] = dataclasses.replace(span, offset=start, length=length,
                                                     transform_descriptors=["stream.zstd/1"])
                counter += 1
                if counter % 100 == 0:
                    atomic_json(temp / "compression_status.json",
                                dict(stage="compressing", items=counter, total=len(old.storage_data),
                                     elapsed_seconds=time.time()-begin, free_bytes=floor_bytes(original, floor_gib)))
            output.flush()
            os.fsync(output.fileno())
    new = dataclasses.replace(old, storage_data=new_info)
    with (dest / ".metadata").open("wb") as f:
        pickle.dump(new, f)
        f.flush()
        os.fsync(f.fileno())
    for p in original.iterdir():
        if p.is_file():
            shutil.copy2(p, temp / p.name)
    if (original / "distributed/.metadata").read_bytes() != old_blob:
        raise ValueError("source checkpoint changed during compression")
    atomic_json(temp / "compression_status.json", dict(stage="verifying", items=counter, elapsed_seconds=time.time()-begin))
    proof = verify_pair(original, temp)
    proof.update(storage_format="native_stream.zstd/1_lossless", optimizer_precision_changed=False,
                 elapsed_seconds=time.time()-begin, original_metadata_sha256=hashlib.sha256(old_blob).hexdigest(),
                 compressed_metadata_sha256=hashlib.sha256((dest / ".metadata").read_bytes()).hexdigest(),
                 minimum_free_gib=floor_gib, free_bytes=floor_bytes(original, floor_gib))
    atomic_json(temp / "lossless_storage.json", proof)
    atomic_json(temp / "compression_status.json", dict(stage="verified_ready_to_commit", items=counter, elapsed_seconds=time.time()-begin))
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--temp", type=Path, required=True)
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    proof = convert_resume(args.resume, args.temp)
    if args.commit:
        # Exact byte equality was checked before any old directory is removed.
        import importlib.util
        spec = importlib.util.spec_from_file_location("atomic_dcp_commit", Path(__file__).with_name("atomic_training_checkpoint.py"))
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        floor_bytes(args.resume)
        helper.commit_complete_directory(args.temp, args.resume, args.resume.with_name("resume.old"),
                                        required_files=("metadata.json", "distributed/.metadata", "lossless_storage.json"))
    print(json.dumps({k: v for k, v in proof.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
