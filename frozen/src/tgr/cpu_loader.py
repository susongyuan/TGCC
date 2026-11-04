"""Bounded spawn-only CPU sequence loading, with no parent import side effects.

Every worker owns its dataset/adapter/video readers. Models, policies and GPU
objects cannot be sent to workers. The parent retains sampler and collation.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import resource
import signal
import sys
import time
import threading

_DATASET = None
_MAX_RSS = None
_WORKER_CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}


def _metrics():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "Threads:")):
            key, value = line.split()[:2]
            values[key.rstrip(":")] = int(value)
    return dict(pid=os.getpid(), rss_bytes=values.get("VmRSS", 0) * 1024,
                peak_rss_bytes=usage.ru_maxrss * 1024, threads=values.get("Threads", 0),
                cpu_seconds=usage.ru_utime + usage.ru_stime,
                file_descriptors=len(list(Path("/proc/self/fd").iterdir())))


def _cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("CPU sequence worker returned a non-CPU tensor")
    elif isinstance(value, dict):
        for item in value.values():
            _cpu_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _cpu_tree(item)


def _parent_death_guard(parent_pid):
    """Own Linux child only; prevent ignored group signals creating orphans."""
    if sys.platform != "linux":
        raise RuntimeError("audited Linux spawn worker required")
    import ctypes
    library = ctypes.CDLL(None, use_errno=True)
    if os.getppid() != parent_pid or library.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("cannot establish own-worker parent-death guard")
    if os.getppid() != parent_pid:
        os._exit(70)


def _worker_init(spec, ready):
    global _DATASET, _MAX_RSS
    started = time.monotonic()
    # Group SIGTERM/SIGINT belongs to trainer: checkpoint first, then close pool.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _parent_death_guard(spec["parent_pid"])
    os.environ.update(_WORKER_CPU_ENV)
    os.environ.update(spec["cache_environment"])
    owner = Path(__file__).resolve().parents[2]
    os.environ.update(HF_HOME=str(owner / "cache/huggingface"),
                      HF_HUB_CACHE=str(owner / "cache/huggingface/hub"))
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("spawn worker inherited initialized CUDA")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import pyarrow
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    from .data import LazySequenceDataset
    from .geometry_cache import configure, implementation_metadata
    if not configure(spec["cache_environment"]["TGR_GEOMETRY_CACHE_PATH"], implementation_metadata()):
        raise ValueError("CPU worker requires the validated selected production geometry cache")
    _DATASET = LazySequenceDataset(spec["manifest"], spec["checkpoint"], spec["plan"])
    _MAX_RSS = spec["max_worker_rss_bytes"]
    from multiprocessing.util import Finalize
    Finalize(None, _close_readers, exitpriority=10)
    metrics = _metrics()
    if metrics["rss_bytes"] > _MAX_RSS or torch.cuda.is_initialized():
        raise RuntimeError("CPU worker CUDA/RSS admission failed")
    ready.put(dict(pid=os.getpid(), parent_pid=os.getppid(), uid=os.getuid(), init_seconds=time.monotonic() - started, metrics=metrics,
                   dataset_id=id(_DATASET), adapter_id=id(_DATASET.adapter),
                   cpu_environment={key: os.environ.get(key) for key in _WORKER_CPU_ENV},
                   cache_environment=dict(spec["cache_environment"]), cuda_initialized=torch.cuda.is_initialized(),
                   sigterm_ignored=signal.getsignal(signal.SIGTERM) == signal.SIG_IGN,
                   sigint_ignored=signal.getsignal(signal.SIGINT) == signal.SIG_IGN,
                   parent_death_signal="SIGKILL", torch_threads=torch.get_num_threads(),
                   torch_interop_threads=torch.get_num_interop_threads()))


def _ping():
    return os.getpid()


def _shard(position_indices):
    import torch
    started = time.monotonic()
    before = _metrics()
    samples = [(position, _DATASET[index]) for position, index in position_indices]
    _cpu_tree(samples)
    after = _metrics()
    if after["rss_bytes"] > _MAX_RSS or torch.cuda.is_initialized():
        raise RuntimeError("CPU worker CUDA/RSS bound failed")
    return dict(samples=samples, metrics=dict(after, load_seconds=time.monotonic() - started,
                                             load_cpu_seconds=after["cpu_seconds"] - before["cpu_seconds"],
                                             cuda_initialized=torch.cuda.is_initialized()))


def _close_readers():
    if _DATASET is not None:
        _DATASET.close()
    return os.getpid()


def _ordered(shards, count):
    samples = [None] * count
    for shard in shards:
        for position, sample in shard["samples"]:
            if not isinstance(position, int) or not 0 <= position < count or samples[position] is not None:
                raise ValueError("duplicate/invalid CPU sample position")
            _cpu_tree(sample)
            samples[position] = sample
    if any(value is None for value in samples):
        raise ValueError("missing CPU sample position")
    return samples


def _process_identity(pid):
    try:
        path = Path("/proc") / str(pid)
        raw = (path / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return dict(pid=int(pid), uid=path.stat().st_uid, parent_pid=int(fields[1]),
                    start_ticks=int(fields[19]), state=fields[0])
    except (OSError, ValueError, IndexError):
        return None


def _same_owned_worker(recorded, current, parent_pid, uid):
    return bool(recorded and current and current["uid"] == uid
        and current["parent_pid"] == parent_pid
        and all(recorded[key] == current[key] for key in ("pid", "uid", "parent_pid", "start_ticks")))


def _bounded_pool_shutdown(pool, graceful_timeout=10.0, kill_join_timeout=2.0):
    """One total grace deadline, then identity-checked own-child kill and join."""
    if graceful_timeout <= 0 or kill_join_timeout <= 0:
        raise ValueError("positive bounded CPU shutdown deadlines required")
    started = time.monotonic()
    parent_pid, uid = os.getpid(), os.getuid()
    processes = list(dict(getattr(pool, "_processes", None) or {}).values())
    manager = getattr(pool, "_executor_manager_thread", None)
    recorded = {process.pid: _process_identity(process.pid) for process in processes if process.pid}
    report = dict(parent_pid=parent_pid, uid=uid, graceful_timeout_seconds=graceful_timeout,
        kill_join_timeout_seconds=kill_join_timeout, recorded_workers=recorded,
        forced_kill_pids=[], ownership_skipped=[], errors=[])
    # wait=False is essential: ignored SIGTERM/native IO must not hang this call.
    pool.shutdown(wait=False, cancel_futures=True)
    grace_end = started + graceful_timeout
    for process in processes:
        try:
            process.join(timeout=max(0.0, grace_end - time.monotonic()))
        except (AssertionError, OSError) as error:
            report["errors"].append(str(error))
    for process in processes:
        if not process.is_alive():
            continue
        current = _process_identity(process.pid)
        if not _same_owned_worker(recorded.get(process.pid), current, parent_pid, uid):
            report["ownership_skipped"].append(dict(pid=process.pid, current=current))
            continue
        try:
            os.kill(process.pid, signal.SIGKILL)
            report["forced_kill_pids"].append(process.pid)
        except ProcessLookupError:
            pass
        except OSError as error:
            report["errors"].append(str(error))
    final_end = time.monotonic() + kill_join_timeout if report["forced_kill_pids"] else grace_end
    for process in processes:
        try:
            process.join(timeout=max(0.0, final_end - time.monotonic()))
        except (AssertionError, OSError) as error:
            report["errors"].append(str(error))
    if manager is not None and manager is not threading.current_thread():
        manager.join(timeout=max(0.0, final_end - time.monotonic()))
    report.update(elapsed_seconds=time.monotonic() - started,
        alive_worker_pids=[process.pid for process in processes if process.is_alive()],
        exit_codes={process.pid: process.exitcode for process in processes},
        manager_alive=bool(manager and manager.is_alive()))
    report["bounded_close_complete"] = not report["alive_worker_pids"] and not report["manager_alive"]
    report["normal_sentinel_close"] = report["bounded_close_complete"] and not report["forced_kill_pids"]
    return report


class ParallelCPUSequenceLoader:
    """Independent CPU workers. Parent owns order; at most one pending micro<=4.

    cache_environment must come from validated selected_cache_environment(run).
    Only primitive path/environment/index data is submitted to spawned children.
    load(indices) returns the original collated CPU batch; last_load_metrics and
    initialization hold infrastructure evidence. Parent whole-trainer RSS is not
    constrained by the worker-only 6GiB bound.
    """
    def __init__(self, manifest, checkpoint, plan, cache_environment, *, workers=2,
                 max_batch=4, max_worker_rss_gib=6, startup_timeout=120, load_timeout=300,
                 shutdown_timeout=10.0, kill_join_timeout=2.0):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("construct and prestart CPU pool in the trainer main thread")
        if workers not in (2, 4) or not 1 <= max_batch <= 4 or max_worker_rss_gib <= 0:
            raise ValueError("two or four workers, micro<=4 and positive worker RSS bound required")
        paths = (manifest, checkpoint, plan)
        if any(not isinstance(value, (str, os.PathLike)) for value in paths):
            raise TypeError("only manifest/checkpoint/plan paths; models/policies forbidden")
        keys = {"TGR_GEOMETRY_CACHE_PATH", "TGR_GEOMETRY_CACHE_BYPASS", "TGR_GEOMETRY_CACHE_ALLOW_PILOT"}
        if not isinstance(cache_environment, dict) or set(cache_environment) != keys:
            raise ValueError("validated selected-cache environment dictionary required")
        if (cache_environment["TGR_GEOMETRY_CACHE_BYPASS"] != "0"
                or cache_environment["TGR_GEOMETRY_CACHE_ALLOW_PILOT"] != "0"
                or not isinstance(cache_environment["TGR_GEOMETRY_CACHE_PATH"], str)
                or not cache_environment["TGR_GEOMETRY_CACHE_PATH"]):
            raise ValueError("selected complete production cache required, no fresh/pilot path")
        self._lock = threading.RLock()
        self.max_batch = max_batch
        self.workers = workers
        self.load_timeout = load_timeout
        self.shutdown_timeout = shutdown_timeout
        self.kill_join_timeout = kill_join_timeout
        self.shutdown_report = None
        self.closed = False
        self.pending = None
        self.last_load_metrics = None
        self.context = mp.get_context("spawn")
        self.ready = self.context.Queue()
        spec = dict(manifest=str(manifest), checkpoint=str(checkpoint), plan=str(plan),
                    cache_environment=dict(cache_environment), parent_pid=os.getpid(),
                    max_worker_rss_bytes=int(max_worker_rss_gib * 1024**3))
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=self.context,
                                       initializer=_worker_init, initargs=(spec, self.ready))
        started = time.monotonic()
        futures = [self.pool.submit(_ping) for _ in range(workers)]
        try:
            startup_deadline = started + startup_timeout
            self.initialization = [self.ready.get(timeout=max(0.0, startup_deadline - time.monotonic())) for _ in range(workers)]
            for future in futures:
                future.result(timeout=max(0.0, startup_deadline - time.monotonic()))
            if len({item["pid"] for item in self.initialization}) != workers:
                raise RuntimeError("independent initialized CPU workers required")
        except BaseException:
            self.close()
            raise
        self.cold_init_seconds = time.monotonic() - started

    def load(self, indices):
        if self.closed:
            raise RuntimeError("CPU loader is closed")
        if self.pending is not None:
            raise ValueError("one pending CPU batch maximum")
        indices = list(indices)
        if (not 1 <= len(indices) <= self.max_batch
                or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)):
            raise ValueError("ordered integer sample indices, micro<=4 required")
        middle = (len(indices) + 1) // 2
        chunks = ([list(enumerate(indices))[:middle], list(enumerate(indices))[middle:]]
                  if self.workers == 2 else [[(position, index)] for position, index in enumerate(indices)])
        started = time.monotonic()
        if not self._lock.acquire(blocking=False):
            raise ValueError("one pending CPU batch maximum")
        try:
            if self.closed or self.pending is not None:
                raise RuntimeError("closed or occupied CPU loader")
            self.pending = [self.pool.submit(_shard, chunk) for chunk in chunks if chunk]
            load_deadline = started + self.load_timeout
            shards = [future.result(timeout=max(0.0, load_deadline - time.monotonic())) for future in self.pending]
            from .data import collate_sequences
            collate_start = time.monotonic()
            batch = collate_sequences(_ordered(shards, len(indices)))
            _cpu_tree(batch)
            self.last_load_metrics = dict(load_ipc_collate_seconds=time.monotonic() - started,
                parent_collate_seconds=time.monotonic() - collate_start,
                workers=[item["metrics"] for item in shards], parent=_metrics())
            return batch
        except BaseException:
            self.close()
            raise
        finally:
            self.pending = None
            self._lock.release()

    def close(self):
        with self._lock:
            if self.closed:
                return self.shutdown_report
            self.closed = True
            # Normal sentinels get one grace window. Stuck ignored-signal workers
            # are killed only after PID/start-time/UID/parent identity verification.
            try:
                self.shutdown_report = _bounded_pool_shutdown(self.pool,
                    self.shutdown_timeout, self.kill_join_timeout)
            finally:
                self.pending = None
                self.ready.close()
            if not self.shutdown_report["bounded_close_complete"]:
                raise RuntimeError("CPU pipeline bounded shutdown incomplete: " + repr(self.shutdown_report))
            return self.shutdown_report

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
