# GPU / ML Performance Validation Platform

A harness that answers three questions about a PyTorch model on a given device:

1. **Is it still correct?** Each dtype variant (fp32 / bf16 / fp16) is compared against an fp32 reference with a scaled-error metric, cosine similarity and top-1 agreement; batch-size invariance and bitwise run-to-run determinism are also checked.
2. **Is it fast enough?** Latency distribution (p50/p95/p99, coefficient of variation) measured with CUDA events on GPU and `perf_counter` on CPU, with warm-up. Declarative targets (`max_p99_ms`, `min_throughput_per_s`, `max_peak_mem_mb`, `min_speedup_vs_fp32`) turn this into pass/fail, and `mlperf compare` flags p50 regressions between two runs for CI.
3. **How close to the hardware limit?** FLOPs are counted analytically (`FlopCounterMode`), minimum memory traffic is computed from tensor sizes, and both are divided by ceilings **measured on the same machine** by small C++/CUDA micro-benchmarks (`cpp/`). `torch.profiler` lists the top operators / GPU kernels.

Python 3 + PyTorch, C++ (OpenMP) and CUDA for the ceilings, pytest for the test suite, Docker for packaging.

## Quick start

```bash
pip install -r requirements.txt
make -C cpp                      # CPU ceilings (add `make -C cpp gpu` with nvcc for GPU ceilings)
python -m mlperf run --config configs/cpu_smoke.yaml
python -m mlperf run --config configs/gpu.yaml --json results/gpu.json --report results/gpu.md
python -m mlperf compare results/baseline.json results/gpu.json --max-regression-pct 10   # exit 1 on regression
python -m pytest -q
```

`run` exits non-zero if any validation or target fails, or if a target matches no measured variant (a typo must not pass silently).

## Layout

| path | role |
|---|---|
| `mlperf/models.py` | deterministic MLP / ConvNet / Transformer-encoder subjects, `tiny` and `base` scales |
| `mlperf/validate.py` | scaled error, cosine, top-1, tolerances per dtype (overridable in config) |
| `mlperf/timing.py`, `stats.py` | CUDA-event / wall-clock timing, percentile summary |
| `mlperf/profiling.py` | profiler top ops, FLOP count, minimum traffic |
| `mlperf/targets.py` | pass/fail rules, regression comparison |
| `mlperf/ceilings.py`, `cpp/` | measured memory-bandwidth and fp32 ceilings (OpenMP triad + read sum + FMA loop; CUDA copy + FMA kernels) |
| `mlperf/runner.py`, `report.py`, `cli.py` | orchestration, Markdown report, CLI |
| `tests/test_core.py` | 27 tests (unit + a real end-to-end CPU run + CLI exit codes) |

## Measured results (CPU, base-size models)

Run in a 2-vCPU cloud VM (AVX-512 + AMX capable CPU, PyTorch 2.15 CPU). **This is a CPU run; no GPU was available when this was built.**
Config: `configs/cpu_base.yaml`, 20 timed iterations after 5 warm-up. Ceilings measured in the same run: ~27 GB/s memory, ~207 GFLOP/s fp32.

| model | dtype | batch | valid | p50 ms | p99 ms | cv | items/s | GFLOP/s | % fp32 peak | eff. GB/s (lower bound) | % BW peak | peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | fp32 | 1 | ok | 2.395 | 2.661 | 0.04 | 418 | 20.9 | 10.1 | 41.89 | 155.3 | - |
| mlp | fp32 | 16 | ok | 9.735 | 11.282 | 0.07 | 1644 | 82.4 | 39.8 | 10.32 | 38.3 | - |
| mlp | bf16 | 1 | ok | 1.463 | 1.557 | 0.04 | 684 | 34.3 | 16.6 | 34.30 | 127.2 | - |
| mlp | bf16 | 16 | ok | 2.753 | 3.396 | 0.08 | 5812 | 291.4 | 140.8 | 18.25 | 67.7 | - |
| convnet | fp32 | 1 | ok | 3.905 | 5.421 | 0.10 | 256 | 139.0 | 67.2 | 1.21 | 4.5 | - |
| convnet | fp32 | 16 | ok | 46.988 | 74.803 | 0.18 | 341 | 184.8 | 89.3 | 0.12 | 0.4 | - |
| convnet | bf16 | 1 | ok | 2.778 | 3.243 | 0.10 | 360 | 195.4 | 94.4 | 0.85 | 3.2 | - |
| convnet | bf16 | 16 | ok | 18.131 | 31.595 | 0.20 | 882 | 478.9 | 231.5 | 0.15 | 0.6 | - |
| transformer | fp32 | 1 | ok | 6.218 | 11.862 | 0.26 | 161 | 67.5 | 32.6 | 2.06 | 7.6 | - |
| transformer | fp32 | 16 | ok | 31.606 | 51.629 | 0.16 | 506 | 212.4 | 102.6 | 0.44 | 1.6 | - |
| transformer | bf16 | 1 | ok | 2.702 | 3.197 | 0.07 | 370 | 155.3 | 75.0 | 2.37 | 8.8 | - |
| transformer | bf16 | 16 | ok | 12.747 | 21.305 | 0.19 | 1255 | 526.5 | 254.5 | 0.54 | 2.0 | - |

How to read this:
- All 12 variants pass validation against fp32 (bf16 within 8e-2 scaled error / 0.995 cosine), are batch-invariant and bitwise deterministic.
- bf16 is about 1.4x-3.5x faster than fp32 here (p50), and runs above the fp32 ceiling because this CPU has matrix units that a plain fp32 FMA loop cannot use. That is a limit of the ceiling, not a bug.
- `mlp fp32 batch 1` reports 155% of the bandwidth ceiling: the measured ceiling (2 threads, simple loops) understates what the BLAS library gets from this VM. Efficiency percentages are indicative, and the report footer says so.
- Ceilings fluctuate run to run on a shared VM (fp32 ~206-263 GFLOP/s across repeats), and p99 on a noisy VM is high (cv up to 0.26). Treat differences of ~10-20% as noise.

## Status of each part

| part | status |
|---|---|
| validation, timing, targets, profiling, report, CLI, CPU ceilings | built and tested here (27 passing tests, real CPU runs) |
| CUDA code paths (`--device cuda`, CUDA-event timing, fp16, GPU profiler kernel ranking) | written but **not executed** - no GPU was available. Run `configs/gpu.yaml` on a GPU (e.g. a free Colab T4) and treat the first run as the test |
| `cpp/ceilings_gpu.cu` | compiled to PTX with clang for sm_75 as a syntax/codegen check only; not run |
| `Dockerfile` | written but **not built** (no Docker daemon available when this was made) |

## Limitations

- FLOP counting covers matmul/conv/attention ops only; elementwise ops are not counted.
- "Effective GB/s" uses minimum traffic (parameters + input + output); real traffic is higher, so it is a lower bound.
- Models are small stand-ins for real workloads; results say nothing about specific production models.
- Targets in `configs/` are sanity bounds for a small VM, not performance claims.
