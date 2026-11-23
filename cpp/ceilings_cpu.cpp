// Measures what this CPU can actually sustain, so benchmark results can be reported as a fraction of it.
//   mem_bw_gbs  : max of a STREAM-style triad (a=b+s*c) and a read-only sum, over arrays far larger than cache
//   fp32_gflops : independent FMA chains in registers (no memory traffic), all threads
// Prints one JSON object. Build: g++ -O3 -march=native -fopenmp ceilings_cpu.cpp -o ceilings_cpu
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double triad_gbs() {
    const size_t n = 1u << 25;  // 32M floats = 128 MB per array, well beyond any LLC
    std::vector<float> a(n), b(n), c(n);
#pragma omp parallel for
    for (long i = 0; i < (long)n; ++i) { a[i] = 0; b[i] = 1.f; c[i] = 2.f; }  // first-touch
    double best = 1e30;
    for (int r = 0; r < 7; ++r) {
        double t0 = now();
#pragma omp parallel for
        for (long i = 0; i < (long)n; ++i) a[i] = b[i] + 3.0f * c[i];
        best = std::min(best, now() - t0);
    }
    if (a[n / 2] != 7.0f) { std::fprintf(stderr, "triad verification failed\n"); std::exit(1); }
    return 3.0 * n * sizeof(float) / best / 1e9;  // write-allocate traffic not counted (STREAM convention)
}

static double fma_gflops() {
    constexpr int CHAINS = 128, ITERS = 1 << 21;  // 128 floats = 8 AVX-512 / 16 AVX2 independent vector chains: hides FMA latency, stays in registers
    double best_rate = 0;
    float sink = 0;
    for (int r = 0; r < 12; ++r) {
        double t0 = now();
#pragma omp parallel reduction(+ : sink)
        {
            float v[CHAINS];
            for (int k = 0; k < CHAINS; ++k) v[k] = 1.0f + 1e-3f * k + omp_get_thread_num();
            const float m = 0.9999999f, c = 1e-7f;
            for (int i = 0; i < ITERS; ++i)
#pragma omp simd
                for (int k = 0; k < CHAINS; ++k) v[k] = v[k] * m + c;
            for (int k = 0; k < CHAINS; ++k) sink += v[k];
        }
        double dt = now() - t0;
        double flops = 2.0 * CHAINS * (double)ITERS * omp_get_max_threads();
        best_rate = std::max(best_rate, flops / dt / 1e9);
    }
    if (sink == 12345.678f) std::puts("");  // keep the result live
    return best_rate;
}

// Read-only streaming sum: the access pattern of reading model weights once (no write traffic).
static double read_gbs() {
    const size_t n = 1u << 26;  // 256 MB
    std::vector<float> a(n, 1.0f);
    double best = 1e30;
    float tot = 0;
    for (int r = 0; r < 7; ++r) {
        double t0 = now();
        float s = 0;
#pragma omp parallel for simd reduction(+ : s)
        for (long i = 0; i < (long)n; ++i) s += a[i];
        best = std::min(best, now() - t0);
        tot += s;
    }
    if (tot < 0) std::puts("");
    return n * sizeof(float) / best / 1e9;
}

int main() {
    double bw = std::max(triad_gbs(), read_gbs()), gf = fma_gflops();
    std::printf("{\"device\":\"cpu\",\"threads\":%d,\"mem_bw_gbs\":%.2f,\"fp32_gflops\":%.2f}\n", omp_get_max_threads(), bw, gf);
}
