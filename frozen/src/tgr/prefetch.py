"""Single bounded CPU batch prefetch. No GPU tensors or model state in worker."""

from concurrent.futures import ThreadPoolExecutor
import time


class CPUPrefetch:
    def __init__(self, loader):
        self.loader = loader
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def _load(self, rows):
        import torch

        t = time.monotonic()
        batch = self.loader(rows)

        def check(value):
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise ValueError("prefetch must remain CPU-only")
            if isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    check(item)

        check(batch)
        return batch, time.monotonic() - t

    def submit(self, key, rows):
        if self.pending is not None:
            raise ValueError("one pending CPU minibatch maximum")
        self.pending = (key, self.pool.submit(self._load, list(rows)))

    def take(self, key, rows):
        if self.pending is None:
            self.submit(key, rows)
        expected, future = self.pending
        if expected != key:
            raise ValueError("prefetch order mismatch")
        self.pending = None
        return future.result()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending = None
