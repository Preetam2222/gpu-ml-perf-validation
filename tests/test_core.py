import json, math
import pytest
import torch

from mlperf import ceilings, models, profiling, report, runner, stats, targets, timing, validate


# ---------- stats ----------
def test_summarize_known_values():
    s = stats.summarize([1, 2, 3, 4, 5])
    assert s["median"] == 3 and s["min"] == 1 and s["max"] == 5 and s["n"] == 5
    assert s["mean"] == 3 and math.isclose(s["std"], math.sqrt(2.5))

def test_summarize_single_and_empty():
    assert stats.summarize([7.0])["std"] == 0.0
    with pytest.raises(ValueError):
        stats.summarize([])

def test_percentile_interpolates():
    assert stats.percentile([0.0, 10.0], 50) == 5.0
    assert stats.percentile([1, 2, 3, 4], 100) == 4


# ---------- validate ----------
def test_compare_identical_is_perfect():
    t = torch.randn(4, 10)
    m = validate.compare_outputs(t, t.clone())
    assert m["max_abs_err"] == 0 and math.isclose(m["cosine"], 1.0) and m["top1_agreement"] == 1.0

def test_nan_output_fails():
    ref = torch.randn(2, 5)
    bad = ref.clone(); bad[0, 0] = float("nan")
    r = validate.validate_variant(ref, bad, "fp32")
    assert not r.passed and "NaN" in r.failures[0]

def test_scaled_error_is_robust_near_zero():
    ref = torch.tensor([[1e-9, 1.0, -1.0]])
    out = torch.tensor([[2e-9, 1.0, -1.0]])  # huge relative error on a near-zero element only
    assert validate.validate_variant(ref, out, "fp32").passed

def test_wrong_output_fails_and_shape_mismatch_raises():
    ref = torch.randn(3, 8)
    assert not validate.validate_variant(ref, ref + 1.0, "bf16").passed
    with pytest.raises(ValueError):
        validate.compare_outputs(ref, ref[:2])

def test_tolerance_override():
    t = validate.tolerances_for("fp16", {"fp16": {"max_scaled_err": 0.5}})
    assert t["max_scaled_err"] == 0.5 and t["min_cosine"] == validate.DEFAULT_TOLERANCES["fp16"]["min_cosine"]


# ---------- models ----------
@pytest.mark.parametrize("name", sorted(models.MODELS))
def test_models_deterministic_and_shaped(name):
    a, b = models.build(name, "tiny"), models.build(name, "tiny")
    x = models.make_input(a, 3)
    assert x.shape == (3, *a.input_shape)
    with torch.no_grad():
        assert torch.equal(a.module(x), b.module(x))

def test_unknown_model_and_scale():
    with pytest.raises(KeyError):
        models.build("nope")
    with pytest.raises(ValueError):
        models.build("mlp", "huge")


# ---------- timing ----------
def test_timing_counts_calls():
    n = {"c": 0}
    def f(): n["c"] += 1
    out = timing.time_callable(f, torch.device("cpu"), warmup=3, iters=5)
    assert n["c"] == 8 and len(out) == 5 and all(v >= 0 for v in out)

def test_timing_rejects_bad_args():
    with pytest.raises(ValueError):
        timing.time_callable(lambda: None, torch.device("cpu"), iters=0)


# ---------- profiling ----------
def test_flops_of_linear_exact():
    lin = torch.nn.Linear(16, 8, bias=False)
    assert profiling.count_flops(lin, torch.randn(4, 16)) == 2 * 4 * 16 * 8

def test_min_traffic_bytes():
    lin = torch.nn.Linear(4, 2, bias=False)
    x, y = torch.zeros(3, 4), torch.zeros(3, 2)
    assert profiling.min_traffic_bytes(lin, x, y) == (8 + 12 + 6) * 4

def test_top_ops_cpu_percentages():
    lin = torch.nn.Linear(64, 64)
    x = torch.randn(8, 64)
    ops = profiling.top_ops(lambda: lin(x), torch.device("cpu"))
    assert ops and sum(o["pct"] for o in ops) <= 100.001


# ---------- targets ----------
def _res(model="mlp", dtype="fp32", batch=1, p50=1.0, p99=2.0, tput=1000.0, mem=None):
    return {"model": model, "dtype": dtype, "batch": batch, "status": "ok", "throughput_per_s": tput,
            "peak_mem_mb": mem, "latency_ms": {"median": p50, "p99": p99}}

def test_targets_pass_fail_nomatch():
    rs = [_res(p50=1.0), _res(model="cnn", p50=9.0)]
    out = targets.evaluate([
        {"name": "a", "match": {"model": "mlp"}, "max_p50_ms": 2},
        {"name": "b", "match": {"model": "cnn"}, "max_p50_ms": 2},
        {"name": "c", "match": {"model": "xxx"}, "max_p50_ms": 2},
    ], rs)
    assert [o.status for o in out] == ["pass", "fail", "no_match"]

def test_speedup_vs_fp32():
    rs = [_res(dtype="fp32", tput=100), _res(dtype="fp16", tput=150)]
    ok = targets.evaluate([{"match": {"dtype": "fp16"}, "min_speedup_vs_fp32": 1.4}], rs)[0]
    bad = targets.evaluate([{"match": {"dtype": "fp16"}, "min_speedup_vs_fp32": 2.0}], rs)[0]
    assert ok.status == "pass" and bad.status == "fail"

def test_memory_target_on_cpu_fails_loudly():
    out = targets.evaluate([{"match": {}, "max_peak_mem_mb": 10}], [_res(mem=None)])[0]
    assert out.status == "fail" and "not measurable" in out.details[0]

def test_rule_validation():
    for bad in ([{"max_p50_ms": 1}], [{"match": {}}], [{"match": {}, "max_p50_ms": 1, "typo": 2}]):
        with pytest.raises(ValueError):
            targets.validate_rules(bad)

def test_regressions():
    base = [_res(p50=10.0)]
    rows = targets.find_regressions(base, [_res(p50=12.0)], 10.0)
    assert rows[0]["regressed"] and math.isclose(rows[0]["change_pct"], 20.0)
    assert not targets.find_regressions(base, [_res(p50=10.5)], 10.0)[0]["regressed"]
    assert targets.find_regressions(base, [_res(model="other")], 10.0) == []


# ---------- ceilings ----------
def test_ceilings_parse():
    d = ceilings.parse('{"device":"cpu","mem_bw_gbs":10.5,"fp32_gflops":40}')
    assert d["mem_bw_gbs"] == 10.5
    with pytest.raises(ValueError):
        ceilings.parse('{"device":"cpu","mem_bw_gbs":10}')
    with pytest.raises(ValueError):
        ceilings.parse('{"device":"cpu","mem_bw_gbs":0,"fp32_gflops":1}')


# ---------- runner / config ----------
def test_merge_config_rules():
    with pytest.raises(ValueError):
        runner.merge_config({"bogus": 1})
    with pytest.raises(ValueError):
        runner.merge_config({"matrix": {"models": ["nope"]}})
    with pytest.raises(ValueError):
        runner.merge_config({"matrix": {"dtypes": ["bf16"]}})  # fp32 reference required
    assert runner.merge_config({"scale": "tiny"})["matrix"]["batches"] == [1, 32]

def test_end_to_end_cpu_tiny_and_report():
    cfg = runner.merge_config({"device": "cpu", "scale": "tiny", "timing": {"warmup": 1, "iters": 5},
                               "matrix": {"models": ["mlp", "transformer"], "dtypes": ["fp32", "bf16", "fp16"], "batches": [1, 4]},
                               "targets": [{"name": "t", "match": {"model": "mlp"}, "max_p99_ms": 1000}]})
    res = runner.run(cfg)
    assert res["summary"]["passed"], res["summary"]
    skipped = [r for r in res["results"] if r["status"] == "skipped"]
    assert skipped and all(r["dtype"] == "fp16" for r in skipped)  # fp16 on CPU is skipped, not faked
    ok = [r for r in res["results"] if r["status"] == "ok"]
    assert all(r["validation"]["metrics"]["bitwise_deterministic"] == 1.0 for r in ok)
    md = report.to_markdown(res)
    assert "PASS" in md and "| mlp |" in md
    json.dumps(res)  # results must be serializable

def test_failing_validation_is_reported():
    cfg = runner.merge_config({"device": "cpu", "scale": "tiny", "timing": {"warmup": 0, "iters": 2},
                               "matrix": {"models": ["mlp"], "dtypes": ["fp32", "bf16"], "batches": [2]},
                               "validation": {"bf16": {"max_scaled_err": 1e-12}}})
    res = runner.run(cfg)
    assert res["summary"]["validation_failures"] == 1 and not res["summary"]["passed"]

def test_cli_exit_codes(tmp_path):
    from mlperf import cli
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text("device: cpu\nscale: tiny\ntiming: {warmup: 0, iters: 2}\nmatrix: {models: [mlp], dtypes: [fp32], batches: [2]}\n")
    out = tmp_path / "r.json"
    assert cli.main(["run", "--config", str(cfgp), "--json", str(out)]) == 0
    assert cli.main(["compare", str(out), str(out)]) == 0
    slow = json.loads(out.read_text())
    for r in slow["results"]:
        r["latency_ms"]["median"] *= 2
    slowp = tmp_path / "slow.json"; slowp.write_text(json.dumps(slow))
    assert cli.main(["compare", str(out), str(slowp), "--max-regression-pct", "10"]) == 1
