"""Functional validation: does a model variant (lower precision, different batch size) still
compute the same function as the fp32 reference?

The error metric is scaled by the reference's magnitude (max|out - ref| / max|ref|) instead of
dividing element-wise by |ref|, which blows up when reference values are near zero.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

DEFAULT_TOLERANCES: Dict[str, Dict[str, float]] = {
    "fp32": {"max_scaled_err": 1e-4, "min_cosine": 0.99999},
    "bf16": {"max_scaled_err": 8e-2, "min_cosine": 0.995},
    "fp16": {"max_scaled_err": 2e-2, "min_cosine": 0.999},
}


@dataclass
class ValidationResult:
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    tolerances: Dict[str, float] = field(default_factory=dict)
    failures: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "metrics": self.metrics, "tolerances": self.tolerances,
                "failures": self.failures}


def compare_outputs(ref: torch.Tensor, out: torch.Tensor) -> Dict[str, float]:
    """Numeric agreement metrics between a reference output and a candidate output."""
    if ref.shape != out.shape:
        raise ValueError(f"shape mismatch: {tuple(ref.shape)} vs {tuple(out.shape)}")
    r, o = ref.detach().double().flatten(), out.detach().double().flatten()
    finite = bool(torch.isfinite(o).all())
    if not finite:
        return {"finite": 0.0, "max_abs_err": float("inf"), "max_scaled_err": float("inf"), "cosine": 0.0}
    diff = (o - r).abs()
    scale = float(r.abs().max()) + 1e-12
    denom = float(r.norm() * o.norm())
    cosine = float((r @ o) / denom) if denom > 0 else (1.0 if float(diff.max()) == 0 else 0.0)
    metrics = {"finite": 1.0, "max_abs_err": float(diff.max()), "max_scaled_err": float(diff.max()) / scale,
               "cosine": cosine}
    if ref.ndim == 2 and ref.shape[1] > 1:  # classifier logits: how often does the top class agree?
        metrics["top1_agreement"] = float((ref.argmax(1) == out.argmax(1)).double().mean())
    return metrics


def check(metrics: Dict[str, float], tolerances: Dict[str, float]) -> ValidationResult:
    """Apply tolerances: max_scaled_err (upper bound), min_cosine and optional min_top1 (lower bounds)."""
    failures = []
    if metrics.get("finite", 1.0) < 1.0:
        failures.append("output contains NaN/Inf")
    else:
        if metrics["max_scaled_err"] > tolerances["max_scaled_err"]:
            failures.append(f"max_scaled_err {metrics['max_scaled_err']:.3e} > {tolerances['max_scaled_err']:.3e}")
        if metrics["cosine"] < tolerances["min_cosine"]:
            failures.append(f"cosine {metrics['cosine']:.6f} < {tolerances['min_cosine']}")
        if "min_top1" in tolerances and "top1_agreement" in metrics and metrics["top1_agreement"] < tolerances["min_top1"]:
            failures.append(f"top1_agreement {metrics['top1_agreement']:.3f} < {tolerances['min_top1']}")
    return ValidationResult(not failures, dict(metrics), dict(tolerances), failures)


def tolerances_for(dtype: str, overrides: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    tol = dict(DEFAULT_TOLERANCES[dtype])
    if overrides and dtype in overrides:
        tol.update(overrides[dtype])
    return tol


def validate_variant(ref_out: torch.Tensor, out: torch.Tensor, dtype: str,
                     overrides: Optional[Dict[str, Dict[str, float]]] = None) -> ValidationResult:
    return check(compare_outputs(ref_out, out), tolerances_for(dtype, overrides))
