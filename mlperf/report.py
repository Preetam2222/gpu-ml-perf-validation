"""Render a run (dict from runner.run) as Markdown, and compare two runs for regressions."""
from typing import Dict, List

from . import targets


def _f(v, spec=".2f", dash="-"):
    return dash if v is None else format(v, spec)


def to_markdown(run: Dict) -> str:
    env, s = run["environment"], run["summary"]
    L: List[str] = ["# Performance validation report", ""]
    L.append("**Environment:** " + ", ".join(f"{k}={v}" for k, v in env.items()))
    c = run.get("ceilings")
    if c:
        L.append(f"**Measured ceilings ({c['device']}):** {c['mem_bw_gbs']:.1f} GB/s memory, {c['fp32_gflops']:.1f} GFLOP/s fp32")
    else:
        L.append("**Measured ceilings:** not available (build cpp/ to enable efficiency columns)")
    L += ["", f"**Result: {'PASS' if s['passed'] else 'FAIL'}** — {s['variants_run']} variants run, {s['variants_skipped']} skipped, "
          f"{s['validation_failures']} validation failures, {s['target_failures']} target failures, "
          f"{s['targets_without_match']} targets matched nothing.", "",
          "## Results", "",
          "| model | dtype | batch | valid | p50 ms | p99 ms | cv | items/s | GFLOP/s | % fp32 peak | eff. GB/s (lower bound) | % BW peak | peak MB |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in run["results"]:
        if r["status"] != "ok":
            L.append(f"| {r['model']} | {r['dtype']} | {r['batch']} | skipped: {r['skip_reason']} |" + " |" * 9)
            continue
        l = r["latency_ms"]
        L.append(f"| {r['model']} | {r['dtype']} | {r['batch']} | {'ok' if r['validation']['passed'] else 'FAIL'} | "
                 f"{l['median']:.3f} | {l['p99']:.3f} | {l['cv']:.2f} | {r['throughput_per_s']:.0f} | "
                 f"{_f(r['gflops_per_s'], '.1f')} | {_f(r.get('pct_of_measured_flops_ceiling'), '.1f')} | "
                 f"{r['effective_gbs_lower_bound']:.2f} | {_f(r.get('pct_of_measured_bw_ceiling'), '.1f')} | "
                 f"{_f(r['peak_mem_mb'], '.0f')} |")
    fails = [r for r in run["results"] if r["status"] == "ok" and not r["validation"]["passed"]]
    if fails:
        L += ["", "## Validation failures", ""]
        for r in fails:
            for m in r["validation"]["failures"]:
                L.append(f"- {r['model']}/{r['dtype']}/b{r['batch']}: {m}")
    if run["targets"]:
        L += ["", "## Targets", ""]
        for t in run["targets"]:
            L.append(f"- **{t['status'].upper()}** {t['name']}")
            for d in t["details"]:
                L.append(f"  - {d}")
    L += ["", "## Top operators (largest variant per model)", ""]
    seen = {}
    for r in run["results"]:
        if r["status"] == "ok":
            k = r["model"]
            if k not in seen or r["batch"] >= seen[k]["batch"]:
                seen[k] = r
    for r in seen.values():
        L.append(f"**{r['model']} {r['dtype']} b{r['batch']}**")
        for o in r["top_ops"]:
            L.append(f"- {o['name']}: {o['self_ms']:.3f} ms ({o['pct']:.0f}%) x{o['count']}")
        L.append("")
    L.append("_Efficiency columns divide by ceilings measured on this machine by cpp/ micro-benchmarks. "
             "Effective GB/s uses minimum traffic (params + input + output), so it is a lower bound. "
             "The fp32 ceiling is what a simple FMA loop sustains, not a datasheet peak, and it varies between runs on shared "
             "machines; values above 100% (e.g. bf16 on matrix units, or weights served from cache) mean the ceiling does not "
             "apply to that variant._")
    return "\n".join(L) + "\n"


def regression_markdown(rows: List[Dict], max_pct: float) -> str:
    L = [f"## Regression check (threshold +{max_pct}% p50)", "", "| model | dtype | batch | baseline ms | current ms | change | |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['dtype']} | {r['batch']} | {r['baseline_ms']:.3f} | {r['current_ms']:.3f} | "
                 f"{r['change_pct']:+.1f}% | {'REGRESSED' if r['regressed'] else 'ok'} |")
    return "\n".join(L) + "\n"
