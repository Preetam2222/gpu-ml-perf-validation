"""Measured hardware ceilings from the C++ micro-benchmarks in cpp/.

`ceilings_cpu` (STREAM-style triad + FMA loop) and `ceilings_gpu` (CUDA copy + FMA kernels) each
print one JSON object: {"device", "mem_bw_gbs", "fp32_gflops", ...}. The runner divides achieved
numbers by these to say how close a model gets to what the hardware can actually sustain.
"""
import json
import os
import subprocess
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP = os.path.join(_HERE, "..", "cpp")


def find_binary(kind: str) -> Optional[str]:
    """kind: 'cpu' or 'gpu'. Honors MLPERF_CEILINGS_<KIND> (path), else cpp/ceilings_<kind>."""
    env = os.environ.get(f"MLPERF_CEILINGS_{kind.upper()}")
    path = env or os.path.join(_CPP, f"ceilings_{kind}")
    return path if os.path.isfile(path) and os.access(path, os.X_OK) else None


def parse(output: str) -> dict:
    d = json.loads(output)
    for k in ("device", "mem_bw_gbs", "fp32_gflops"):
        if k not in d:
            raise ValueError(f"ceilings output missing '{k}'")
    if d["mem_bw_gbs"] <= 0 or d["fp32_gflops"] <= 0:
        raise ValueError("ceilings must be positive")
    return d


def measure(kind: str, timeout: float = 120.0) -> Optional[dict]:
    """Run the C++ tool; None if it is not built (callers then omit the efficiency columns)."""
    exe = find_binary(kind)
    if exe is None:
        return None
    p = subprocess.run([exe], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{exe} failed: {p.stderr.strip()}")
    return parse(p.stdout)
