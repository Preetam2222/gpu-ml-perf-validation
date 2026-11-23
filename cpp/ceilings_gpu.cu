// GPU ceilings measured with CUDA events:
//   mem_bw_gbs  : device-to-device copy of a large buffer (read + write bytes counted)
//   fp32_gflops : independent FMA chains per thread, no memory traffic
// Prints one JSON object. Build: nvcc -O3 ceilings_gpu.cu -o ceilings_gpu
#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void copy_k(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) out[i] = in[i];
}

template <int CH>
__global__ void fma_k(float* out, int iters, float m, float c) {
    float v[CH];
#pragma unroll
    for (int k = 0; k < CH; ++k) v[k] = 1.0f + 1e-3f * k + threadIdx.x * 1e-6f;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int k = 0; k < CH; ++k) v[k] = __fmaf_rn(v[k], m, c);
    }
    float s = 0;
#pragma unroll
    for (int k = 0; k < CH; ++k) s += v[k];
    if (s == 123456.f) out[0] = s;  // defeat dead-code elimination
}

int main() {
    int dev = 0;
    CK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

    // ---- bandwidth ----
    const size_t bytes = 512ull << 20;  // 512 MB each way
    const size_t n4 = bytes / sizeof(float4);
    float4 *in, *out;
    CK(cudaMalloc(&in, bytes)); CK(cudaMalloc(&out, bytes));
    CK(cudaMemset(in, 0, bytes));
    int blocks = p.multiProcessorCount * 8;
    copy_k<<<blocks, 256>>>(in, out, n4);  // warm-up
    CK(cudaDeviceSynchronize());
    float best_ms = 1e30f;
    for (int r = 0; r < 10; ++r) {
        CK(cudaEventRecord(a));
        copy_k<<<blocks, 256>>>(in, out, n4);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best_ms) best_ms = ms;
    }
    double bw = 2.0 * bytes / (best_ms / 1e3) / 1e9;

    // ---- fp32 compute ----
    constexpr int CH = 16;
    const int iters = 1 << 16;
    int threads = 256, fblocks = p.multiProcessorCount * 8;
    float* sink; CK(cudaMalloc(&sink, 4));
    fma_k<CH><<<fblocks, threads>>>(sink, 16, 0.999f, 1e-7f);
    CK(cudaDeviceSynchronize());
    float fbest = 1e30f;
    for (int r = 0; r < 5; ++r) {
        CK(cudaEventRecord(a));
        fma_k<CH><<<fblocks, threads>>>(sink, iters, 0.9999999f, 1e-7f);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < fbest) fbest = ms;
    }
    double flops = 2.0 * CH * (double)iters * fblocks * threads;
    double gf = flops / (fbest / 1e3) / 1e9;

    std::printf("{\"device\":\"%s\",\"sm\":\"%d.%d\",\"mem_bw_gbs\":%.2f,\"fp32_gflops\":%.2f}\n", p.name, p.major, p.minor, bw, gf);
    return 0;
}
