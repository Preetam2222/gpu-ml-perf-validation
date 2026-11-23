"""Latency statistics."""
import math
from typing import Dict, List, Sequence


def percentile(sorted_vals: Sequence[float], q: float) -> float:
    """q in [0, 100], linear interpolation between closest ranks (same as numpy's default)."""
    if not sorted_vals:
        raise ValueError("percentile of empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def summarize(samples: Sequence[float]) -> Dict[str, float]:
    """n, mean, median, p95, p99, min, max, std (sample), cv (std/mean)."""
    if not samples:
        raise ValueError("no samples")
    s: List[float] = sorted(float(x) for x in samples)
    n = len(s)
    mean = sum(s) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in s) / (n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "median": percentile(s, 50),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "min": s[0],
        "max": s[-1],
        "std": std,
        "cv": std / mean if mean else 0.0,
    }
