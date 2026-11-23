"""Command line: python -m mlperf run|compare ..."""
import argparse
import json
import sys

import yaml

from . import report, runner, targets


def _load_cfg(path):
    if not path:
        return runner.merge_config({})
    with open(path) as f:
        return runner.merge_config(yaml.safe_load(f) or {})


def cmd_run(a) -> int:
    cfg = _load_cfg(a.config)
    if a.device:
        cfg["device"] = a.device
    if a.scale:
        cfg["scale"] = a.scale
    res = runner.run(cfg, progress=lambda m: print(f"[run] {m}", file=sys.stderr))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
    md = report.to_markdown(res)
    if a.report:
        with open(a.report, "w") as f:
            f.write(md)
    print(md)
    return 0 if res["summary"]["passed"] else 1


def cmd_compare(a) -> int:
    with open(a.baseline) as f:
        b = json.load(f)
    with open(a.current) as f:
        c = json.load(f)
    rows = targets.find_regressions(b["results"], c["results"], a.max_regression_pct)
    print(report.regression_markdown(rows, a.max_regression_pct))
    if not rows:
        print("no overlapping variants to compare", file=sys.stderr)
        return 2
    return 1 if any(r["regressed"] for r in rows) else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mlperf")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="validate + benchmark + profile the matrix in a config")
    r.add_argument("--config")
    r.add_argument("--device", help="auto|cpu|cuda (overrides config)")
    r.add_argument("--scale", choices=["tiny", "base"], help="overrides config")
    r.add_argument("--json", help="write full results JSON here")
    r.add_argument("--report", help="write Markdown report here")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare", help="flag p50 regressions between two result JSONs")
    c.add_argument("baseline")
    c.add_argument("current")
    c.add_argument("--max-regression-pct", type=float, default=10.0)
    c.set_defaults(fn=cmd_compare)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
