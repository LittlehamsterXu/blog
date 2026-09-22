#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(x) do {                                                     \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
        std::cerr << cudaGetErrorString(err) << '\n';                          \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

constexpr int BM = 32, BN = 64, BK = 16;
constexpr int TM = 4,  TN = 4;
constexpr int THREAD_COLS = BN / TN;     // 16
constexpr int THREADS = (BM / TM) * (BN / TN); // 128

// -----------------------------------------------------------------------------
// 1) 4x4 baseline
// -----------------------------------------------------------------------------
__global__ void matmul_4x4_baseline(
    const float* A, const float* B, float* C, int N) {

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tr = tid / THREAD_COLS;
    const int tc = tid % THREAD_COLS;

    const int lr = tr * TM;
    const int lc = tc * TN;

    const int row = blockIdx.y * BM + lr;
    const int col = blockIdx.x * BN + lc;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc[TM][TN] = {};

    for (int tile = 0; tile < N / BK; ++tile) {
#pragma unroll 1
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            int r = idx / BK;
            int k = idx % BK;
            As[r][k] =
                A[(blockIdx.y * BM + r) * N + tile * BK + k];
        }

#pragma unroll 1
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            int k = idx / BN;
            int c = idx % BN;
            Bs[k][c] =
                B[(tile * BK + k) * N + blockIdx.x * BN + c];
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];

#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = As[lr + i][k];

#pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = Bs[k][lc + j];

#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(row + i) * N + col + j] = acc[i][j];
}

// -----------------------------------------------------------------------------
// 2) 4x4 + double buffering + async global->shared copy
// -----------------------------------------------------------------------------
__device__ __forceinline__
void async_load_stage(
    const float* A, const float* B,
    float* As, float* Bs,
    int N, int block_row, int block_col, int k0, int tid) {

    // A tile: 512 floats = 128 x float4 -> 1 copy/thread
    int a = tid * 4;
    int ar = a / BK;
    int ak = a % BK;

    __pipeline_memcpy_async(
        As + ar * BK + ak,
        A + (block_row + ar) * N + k0 + ak,
        16);

    // B tile: 1024 floats = 256 x float4 -> 2 copies/thread
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        int b = (tid + rep * THREADS) * 4;
        int bk = b / BN;
        int bc = b % BN;

        __pipeline_memcpy_async(
            Bs + bk * BN + bc,
            B + (k0 + bk) * N + block_col + bc,
            16);
    }

    __pipeline_commit();
}

__global__ void matmul_4x4_double_buffer(
    const float* A, const float* B, float* C, int N) {

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int tr = tid / THREAD_COLS;
    const int tc = tid % THREAD_COLS;

    const int lr = tr * TM;
    const int lc = tc * TN;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int row = block_row + lr;
    const int col = block_col + lc;

    __shared__ __align__(16) float As[2][BM][BK];
    __shared__ __align__(16) float Bs[2][BK][BN];

    float acc[TM][TN] = {};
    const int tiles = N / BK;

    // preload tile 0
    async_load_stage(
        A, B,
        &As[0][0][0], &Bs[0][0][0],
        N, block_row, block_col, 0, tid);

    __pipeline_wait_prior(0);
    __syncthreads();

    for (int tile = 0; tile < tiles; ++tile) {
        const int read  = tile & 1;
        const int write = read ^ 1;

        // start loading tile k+1
        if (tile + 1 < tiles) {
            async_load_stage(
                A, B,
                &As[write][0][0], &Bs[write][0][0],
                N, block_row, block_col,
                (tile + 1) * BK,
                tid);
        }

        // compute tile k while tile k+1 is in flight
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];

#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = As[read][lr + i][k];

#pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = Bs[read][k][lc + j];

#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];
        }

        if (tile + 1 < tiles) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(row + i) * N + col + j] = acc[i][j];
}

enum class Kernel { Baseline, DoubleBuffer };

void launch(Kernel k, const float* A, const float* B, float* C, int N) {
    dim3 block(32, 4);
    dim3 grid(N / BN, N / BM);

    if (k == Kernel::Baseline)
        matmul_4x4_baseline<<<grid, block>>>(A, B, C, N);
    else
        matmul_4x4_double_buffer<<<grid, block>>>(A, B, C, N);
}

double bench(
    Kernel k,
    const float* A, const float* B, float* C,
    int N, int repeat) {

    constexpr int WARMUP = 100;
    constexpr int ROUNDS = 7;

    for (int i = 0; i < WARMUP; ++i)
        launch(k, A, B, C, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> samples;

    for (int r = 0; r < ROUNDS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i)
            launch(k, A, B, C, N);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms / repeat);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

bool check(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C,
    int N) {

    const int pts[][2] = {
        {0, 0}, {N / 2, N / 2}, {N - 1, N - 1}
    };

    for (auto& p : pts) {
        int r = p[0], c = p[1];
        float ref = 0.0f;

        for (int k = 0; k < N; ++k)
            ref += A[r * N + k] * B[k * N + c];

        float got = C[r * N + c];
        if (std::fabs(got - ref) > 1e-3f + 1e-3f * std::fabs(ref))
            return false;
    }

    return true;
}

void run(
    Kernel k, const char* name,
    const std::vector<float>& hA,
    const std::vector<float>& hB,
    std::vector<float>& hC,
    float* dA, float* dB, float* dC,
    int N, int repeat, size_t bytes) {

    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    double ms = bench(k, dA, dB, dC, N, repeat);
    double gflops =
        2.0 * static_cast<double>(N) * N * N / (ms * 1e6);

    CUDA_CHECK(cudaMemcpy(
        hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    std::cout
        << name << '\n'
        << "  time=" << ms << " ms\n"
        << "  performance=" << gflops << " GFLOPS\n"
        << "  check=" << (check(hA, hB, hC, N) ? "OK" : "FAIL")
        << "\n\n";
}

int main(int argc, char** argv) {
    int N = argc > 1 ? std::atoi(argv[1]) : 2048;
    int repeat = argc > 2 ? std::atoi(argv[2]) : 50;

    if (N <= 0 || N % 64 != 0) {
        std::cerr << "N must be divisible by 64\n";
        return 1;
    }

    size_t elems = static_cast<size_t>(N) * N;
    size_t bytes = elems * sizeof(float);

    std::vector<float> hA(elems), hB(elems), hC(elems);

    for (size_t i = 0; i < elems; ++i) {
        hA[i] = static_cast<float>((i * 13) % 101 - 50) / 50.0f;
        hB[i] = static_cast<float>((i * 17) % 97 - 48) / 48.0f;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    run(Kernel::Baseline, "4x4 baseline",
        hA, hB, hC, dA, dB, dC, N, repeat, bytes);

    run(Kernel::DoubleBuffer, "4x4 double buffer",
        hA, hB, hC, dA, dB, dC, N, repeat, bytes);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
