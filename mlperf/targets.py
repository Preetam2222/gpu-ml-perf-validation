"""Performance targets: declarative pass/fail rules evaluated against measured results.

A rule has a `match` (model / dtype / batch, any subset) and one or more thresholds:
  max_p50_ms, max_p99_ms, min_throughput_per_s, max_peak_mem_mb,
  min_speedup_vs_fp32   (throughput relative to the fp32 variant of the same model and batch)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

THRESHOLDS = ("max_p50_ms", "max_p99_ms", "min_throughput_per_s", "max_peak_mem_mb", "min_speedup_vs_fp32")


@dataclass
class Outcome:
    name: str
    status: str  # "pass" | "fail" | "no_match"
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "details": self.details}


def _matches(result: dict, match: Dict) -> bool:
    return all(result.get(k) == v for k, v in match.items())


def _check_one(result: dict, rule: Dict, results: List[dict]) -> List[str]:
    """Returns a list of failure messages for one result (empty = pass)."""
    tag = f"{result['model']}/{result['dtype']}/b{result['batch']}"
    bad = []
    lat = result["latency_ms"]
    if "max_p50_ms" in rule and lat["median"] > rule["max_p50_ms"]:
        bad.append(f"{tag}: p50 {lat['median']:.3f} ms > {rule['max_p50_ms']} ms")
    if "max_p99_ms" in rule and lat["p99"] > rule["max_p99_ms"]:
        bad.append(f"{tag}: p99 {lat['p99']:.3f} ms > {rule['max_p99_ms']} ms")
    if "min_throughput_per_s" in rule and result["throughput_per_s"] < rule["min_throughput_per_s"]:
        bad.append(f"{tag}: throughput {result['throughput_per_s']:.1f}/s < {rule['min_throughput_per_s']}/s")
    if "max_peak_mem_mb" in rule:
        mem = result.get("peak_mem_mb")
        if mem is None:
            bad.append(f"{tag}: peak memory not measurable on this device")
        elif mem > rule["max_peak_mem_mb"]:
            bad.append(f"{tag}: peak memory {mem:.1f} MB > {rule['max_peak_mem_mb']} MB")
    if "min_speedup_vs_fp32" in rule:
        base = next((r for r in results if r["status"] == "ok" and r["dtype"] == "fp32"
                     and r["model"] == result["model"] and r["batch"] == result["batch"]), None)
        if base is None:
            bad.append(f"{tag}: no fp32 baseline to compute speedup")
        else:
            sp = result["throughput_per_s"] / base["throughput_per_s"]
            if sp < rule["min_speedup_vs_fp32"]:
                bad.append(f"{tag}: speedup vs fp32 {sp:.2f}x < {rule['min_speedup_vs_fp32']}x")
    return bad


def validate_rules(rules: List[Dict]) -> None:
    for i, r in enumerate(rules):
        if "match" not in r or not isinstance(r["match"], dict):
            raise ValueError(f"target #{i} needs a 'match' mapping")
        unknown = set(r) - {"name", "match", *THRESHOLDS}
        if unknown:
            raise ValueError(f"target #{i} has unknown keys: {sorted(unknown)}")
        if not any(t in r for t in THRESHOLDS):
            raise ValueError(f"target #{i} has no threshold ({', '.join(THRESHOLDS)})")


def evaluate(rules: List[Dict], results: List[dict]) -> List[Outcome]:
    """Skipped variants are ignored. A rule that matches nothing is reported as 'no_match' (not a pass)."""
    validate_rules(rules)
    ok = [r for r in results if r["status"] == "ok"]
    outcomes = []
    for i, rule in enumerate(rules):
        name = rule.get("name", f"target #{i}")
        hit = [r for r in ok if _matches(r, rule["match"])]
        if not hit:
            outcomes.append(Outcome(name, "no_match", [f"no measured variant matches {rule['match']}"]))
            continue
        bad = [m for r in hit for m in _check_one(r, rule, ok)]
        outcomes.append(Outcome(name, "fail" if bad else "pass", bad))
    return outcomes


def find_regressions(baseline: List[dict], current: List[dict], max_regression_pct: float) -> List[Dict]:
    """Compare p50 latency of matching variants; return those slower by more than the threshold."""
    key = lambda r: (r["model"], r["dtype"], r["batch"])
    base = {key(r): r for r in baseline if r["status"] == "ok"}
    out = []
    for r in current:
        if r["status"] != "ok" or key(r) not in base:
            continue
        b = base[key(r)]["latency_ms"]["median"]
        c = r["latency_ms"]["median"]
        change = (c - b) / b * 100.0
        out.append({"model": r["model"], "dtype": r["dtype"], "batch": r["batch"], "baseline_ms": b,
                    "current_ms": c, "change_pct": change, "regressed": change > max_regression_pct})
    return out
