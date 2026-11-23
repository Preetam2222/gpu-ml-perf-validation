"""Latency measurement. GPU work is timed with CUDA events (not wall clock) so asynchronous
kernel launches are measured correctly; CPU work uses perf_counter."""
import time
from typing import Callable, List

import torch


def time_callable(fn: Callable[[], object], device: torch.device, warmup: int = 10, iters: int = 50) -> List[float]:
    """Run fn() `warmup` times untimed, then `iters` times timed; return per-call milliseconds."""
    if iters < 1 or warmup < 0:
        raise ValueError("iters must be >= 1 and warmup >= 0")
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize(device)
        return [s.elapsed_time(e) for s, e in zip(starts, ends)]
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out
