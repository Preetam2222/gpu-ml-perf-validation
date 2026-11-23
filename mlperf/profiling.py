"""Where does the time go (top operators / GPU kernels) and how much arithmetic does a forward pass do."""
from typing import Callable, Dict, List, Optional

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode


def _device_time_us(evt) -> float:
    # Attribute was renamed (self_cuda_time_total -> self_device_time_total) across torch versions.
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(evt, attr):
            return float(getattr(evt, attr))
    return 0.0


def top_ops(fn: Callable[[], object], device: torch.device, topk: int = 6) -> List[Dict]:
    """One profiled call. On CUDA, ranks by GPU kernel time (kernel names appear as ops);
    on CPU, ranks by self CPU time. Returns [{name, count, self_ms, pct}]."""
    acts = [ProfilerActivity.CPU]
    on_gpu = device.type == "cuda"
    if on_gpu:
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as prof:
        fn()
        if on_gpu:
            torch.cuda.synchronize(device)
    rows = []
    for e in prof.key_averages():
        t_us = _device_time_us(e) if on_gpu else float(e.self_cpu_time_total)
        if t_us > 0:
            rows.append((e.key, int(e.count), t_us))
    total = sum(r[2] for r in rows) or 1.0
    rows.sort(key=lambda r: r[2], reverse=True)
    return [{"name": n, "count": c, "self_ms": t / 1e3, "pct": 100.0 * t / total} for n, c, t in rows[:topk]]


def count_flops(module: torch.nn.Module, x: torch.Tensor) -> Optional[int]:
    """Analytical FLOPs of one forward pass (matmul / conv / attention ops; elementwise ops are not
    counted). Run with grad enabled so fused inference-only attention fast paths, which the counter
    cannot see, are not taken."""
    try:
        with torch.enable_grad(), FlopCounterMode(display=False) as fc:
            module(x)
        return int(fc.get_total_flops())
    except Exception:  # an op the counter cannot handle: report unknown rather than guess
        return None


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def min_traffic_bytes(module: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    """Lower bound on bytes that must move for one forward pass: read all parameters and the input
    once, write the output once. Real traffic is higher (activations), so achieved bandwidth derived
    from this is a lower bound, not a measurement."""
    return sum(tensor_bytes(p) for p in module.parameters()) + tensor_bytes(x) + tensor_bytes(y)
