"""Orchestrates a run: build -> validate -> time -> profile every (model, dtype, batch) variant."""
import copy
import platform
from typing import Callable, Dict, List, Optional

import torch

from . import ceilings, models, profiling, stats, targets, timing, validate

DEFAULT_CONFIG = {
    "device": "auto",
    "seed": 1234,
    "scale": "base",
    "timing": {"warmup": 10, "iters": 50},
    "matrix": {"models": ["mlp", "convnet", "transformer"], "dtypes": ["fp32"], "batches": [1, 32]},
    "validation": {},
    "targets": [],
}


def merge_config(user: Dict) -> Dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in (user or {}).items():
        if k not in DEFAULT_CONFIG:
            raise ValueError(f"unknown config key {k!r}")
        if isinstance(DEFAULT_CONFIG[k], dict) and isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    for m in cfg["matrix"]["models"]:
        if m not in models.MODELS:
            raise ValueError(f"unknown model {m!r}")
    for d in cfg["matrix"]["dtypes"]:
        if d not in models.DTYPES:
            raise ValueError(f"unknown dtype {d!r}")
    if "fp32" not in cfg["matrix"]["dtypes"]:
        raise ValueError("matrix.dtypes must include fp32 (it is the validation reference)")
    targets.validate_rules(cfg["targets"])
    return cfg


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    return dev


def device_info(device: torch.device) -> Dict:
    info = {"device": device.type, "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform()}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update({"gpu": p.name, "sm": f"{p.major}.{p.minor}", "total_mem_gb": round(p.total_memory / 2**30, 2),
                     "sm_count": p.multi_processor_count, "cuda": torch.version.cuda})
    else:
        info.update({"cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads()})
    return info


def skip_reason(dtype: str, device: torch.device) -> Optional[str]:
    if dtype == "fp16" and device.type == "cpu":
        return "fp16 inference is not supported/meaningful on CPU; use bf16 or run on a GPU"
    return None


def _forward(module: torch.nn.Module, x: torch.Tensor) -> Callable[[], torch.Tensor]:
    def run():
        with torch.inference_mode():
            return module(x)
    return run


def run_variant(ref_module, spec: "models.ModelSpec", dtype: str, batch: int, device: torch.device, cfg: Dict,
                ceil: Optional[dict]) -> Dict:
    spec_in = models.make_input(spec, batch)
    rec: Dict = {"model": spec.name, "dtype": dtype, "batch": batch, "status": "ok"}
    why = skip_reason(dtype, device)
    if why:
        rec.update(status="skipped", skip_reason=why)
        return rec

    torch_dtype = models.DTYPES[dtype]
    module = copy.deepcopy(ref_module).to(device=device, dtype=torch_dtype).eval()
    ref = ref_module  # fp32 reference already on `device`
    x32 = spec_in.to(device)
    x = x32.to(torch_dtype)

    # ---- functional validation against the fp32 reference ----
    with torch.inference_mode():
        y_ref = ref(x32).float().cpu()
        y = module(x).float().cpu()
        y_again = module(x).float().cpu()
        checks = validate.validate_variant(y_ref, y, dtype, cfg["validation"])
        # Batch invariance: running one sample alone must match its row in the batched output.
        if batch > 1:
            y1 = module(x[:1]).float().cpu()
            inv = validate.check(validate.compare_outputs(y[:1], y1), validate.tolerances_for(dtype, cfg["validation"]))
            checks.metrics["batch_invariance_scaled_err"] = inv.metrics["max_scaled_err"]
            if not inv.passed:
                checks.passed = False
                checks.failures.append("batch invariance: " + "; ".join(inv.failures))
        checks.metrics["bitwise_deterministic"] = float(torch.equal(y, y_again))
    rec["validation"] = checks.to_dict()

    # ---- timing ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fn = _forward(module, x)
    lat = stats.summarize(timing.time_callable(fn, device, cfg["timing"]["warmup"], cfg["timing"]["iters"]))
    rec["latency_ms"] = lat
    rec["throughput_per_s"] = batch / (lat["median"] / 1e3)
    rec["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None

    # ---- compute / memory efficiency ----
    flops = profiling.count_flops(module, x)
    rec["flops"] = flops
    rec["gflops_per_s"] = flops / (lat["median"] / 1e3) / 1e9 if flops else None
    rec["min_traffic_bytes"] = profiling.min_traffic_bytes(module, x, y)
    rec["effective_gbs_lower_bound"] = rec["min_traffic_bytes"] / (lat["median"] / 1e3) / 1e9
    if ceil:
        rec["pct_of_measured_flops_ceiling"] = (100.0 * rec["gflops_per_s"] / ceil["fp32_gflops"]) if rec["gflops_per_s"] else None
        rec["pct_of_measured_bw_ceiling"] = 100.0 * rec["effective_gbs_lower_bound"] / ceil["mem_bw_gbs"]
    rec["top_ops"] = profiling.top_ops(fn, device)
    return rec


def run(cfg: Dict, progress: Optional[Callable[[str], None]] = None) -> Dict:
    device = resolve_device(cfg["device"])
    ceil = ceilings.measure("gpu" if device.type == "cuda" else "cpu")
    results: List[Dict] = []
    for name in cfg["matrix"]["models"]:
        spec = models.build(name, cfg["scale"], cfg["seed"])
        ref_module = spec.module.to(device).float().eval()
        for dtype in cfg["matrix"]["dtypes"]:
            for batch in cfg["matrix"]["batches"]:
                if progress:
                    progress(f"{name} {dtype} batch={batch}")
                results.append(run_variant(ref_module, spec, dtype, batch, device, cfg, ceil))
    outcomes = targets.evaluate(cfg["targets"], results)
    val_failed = [r for r in results if r["status"] == "ok" and not r["validation"]["passed"]]
    t_failed = [o for o in outcomes if o.status == "fail"]
    no_match = [o for o in outcomes if o.status == "no_match"]
    return {
        "environment": device_info(device),
        "ceilings": ceil,
        "config": cfg,
        "results": results,
        "targets": [o.to_dict() for o in outcomes],
        "summary": {
            "variants_run": sum(r["status"] == "ok" for r in results),
            "variants_skipped": sum(r["status"] == "skipped" for r in results),
            "validation_failures": len(val_failed),
            "target_failures": len(t_failed),
            "targets_without_match": len(no_match),
            "passed": not val_failed and not t_failed and not no_match,
        },
    }
