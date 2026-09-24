#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace nvcuda;

// ============================================================
// Tensor Core GEMM configuration
// ============================================================

constexpr int CTA_M = 64;
constexpr int CTA_N = 64;
constexpr int CTA_K = 16;

constexpr int WARP_M = 32;
constexpr int WARP_N = 32;
constexpr int WARP_K = 16;

constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;

constexpr int WARPS_M = CTA_M / WARP_M;   // 2
constexpr int WARPS_N = CTA_N / WARP_N;   // 2
constexpr int NUM_WARPS = WARPS_M * WARPS_N; // 4

constexpr int WMMA_THREADS = NUM_WARPS * 32; // 128


#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)


// ============================================================
// Ordinary tiled CUDA GEMM
//
// FP16 input
// FP32 accumulation/output
// ============================================================

constexpr int BASE_TILE = 16;

__global__ void tiled_gemm(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ half As[BASE_TILE][BASE_TILE];
    __shared__ half Bs[BASE_TILE][BASE_TILE];

    const int row =
        blockIdx.y * BASE_TILE + threadIdx.y;

    const int col =
        blockIdx.x * BASE_TILE + threadIdx.x;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BASE_TILE) {

        if (row < M &&
            k0 + threadIdx.x < K) {

            As[threadIdx.y][threadIdx.x] =
                A[row * K + k0 + threadIdx.x];

        } else {
            As[threadIdx.y][threadIdx.x] =
                __float2half(0.0f);
        }

        if (col < N &&
            k0 + threadIdx.y < K) {

            Bs[threadIdx.y][threadIdx.x] =
                B[(k0 + threadIdx.y) * N + col];

        } else {
            Bs[threadIdx.y][threadIdx.x] =
                __float2half(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BASE_TILE; ++k) {
            float a = __half2float(
                As[threadIdx.y][k]);

            float b = __half2float(
                Bs[k][threadIdx.x]);

            acc += a * b;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}


// ============================================================
// WMMA Tensor Core GEMM
//
// CTA:
//      64 x 64 x 16
//
// Warp:
//      32 x 32 x 16
//
// MMA:
//      16 x 16 x 16
// ============================================================

__global__ void wmma_gemm(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K)
{
    // --------------------------------------------------------
    // CTA coordinates
    // --------------------------------------------------------

    const int cta_m =
        blockIdx.y * CTA_M;

    const int cta_n =
        blockIdx.x * CTA_N;


    // --------------------------------------------------------
    // Warp coordinates
    // --------------------------------------------------------

    const int warp_id =
        threadIdx.x / 32;

    const int warp_m_id =
        warp_id / WARPS_N;

    const int warp_n_id =
        warp_id % WARPS_N;


    // --------------------------------------------------------
    // CTA shared-memory tiles
    //
    // A: 64 x 16
    // B: 16 x 64
    // --------------------------------------------------------

    __shared__ half As[CTA_M][CTA_K];
    __shared__ half Bs[CTA_K][CTA_N];


    // --------------------------------------------------------
    // Each warp computes:
    //
    //      32 x 32
    //
    // MMA:
    //
    //      16 x 16
    //
    // Therefore:
    //
    //      2 x 2 accumulator fragments
    // --------------------------------------------------------

    wmma::fragment<
        wmma::accumulator,
        MMA_M,
        MMA_N,
        MMA_K,
        float
    > c_frag[WARP_M / MMA_M]
            [WARP_N / MMA_N];


    // Initialize accumulator fragments

    #pragma unroll
    for (int mi = 0;
         mi < WARP_M / MMA_M;
         ++mi) {

        #pragma unroll
        for (int ni = 0;
             ni < WARP_N / MMA_N;
             ++ni) {

            wmma::fill_fragment(
                c_frag[mi][ni],
                0.0f);
        }
    }


    // ========================================================
    // Walk along K dimension
    //
    // Each CTA iteration handles:
    //
    //      A: 64 x 16
    //      B: 16 x 64
    // ========================================================

    for (int k0 = 0;
         k0 < K;
         k0 += CTA_K) {

        // ====================================================
        // Global -> Shared
        //
        // Cooperative load by all 128 threads
        //
        // For now:
        //
        //      ordinary CUDA loads
        //
        // Later:
        //
        //      cp.async + multi-stage
        // ====================================================

        const int tid = threadIdx.x;


        // ------------------
        // Load A tile
        // ------------------

        constexpr int A_TILE_ELEMS =
            CTA_M * CTA_K;

        for (int idx = tid;
             idx < A_TILE_ELEMS;
             idx += blockDim.x) {

            int r = idx / CTA_K;
            int c = idx % CTA_K;

            int gm = cta_m + r;
            int gk = k0 + c;

            if (gm < M && gk < K) {
                As[r][c] =
                    A[gm * K + gk];
            } else {
                As[r][c] =
                    __float2half(0.0f);
            }
        }


        // ------------------
        // Load B tile
        // ------------------

        constexpr int B_TILE_ELEMS =
            CTA_K * CTA_N;

        for (int idx = tid;
             idx < B_TILE_ELEMS;
             idx += blockDim.x) {

            int r = idx / CTA_N;
            int c = idx % CTA_N;

            int gk = k0 + r;
            int gn = cta_n + c;

            if (gk < K && gn < N) {
                Bs[r][c] =
                    B[gk * N + gn];
            } else {
                Bs[r][c] =
                    __float2half(0.0f);
            }
        }


        __syncthreads();


        // ====================================================
        // Shared -> Warp fragments
        //
        // Each warp:
        //
        //      A fragments: 2
        //      B fragments: 2
        //
        // giving:
        //
        //      2 x 2 = 4 MMA operations
        // ====================================================

        wmma::fragment<
            wmma::matrix_a,
            MMA_M,
            MMA_N,
            MMA_K,
            half,
            wmma::row_major
        > a_frag[WARP_M / MMA_M];


        wmma::fragment<
            wmma::matrix_b,
            MMA_M,
            MMA_N,
            MMA_K,
            half,
            wmma::row_major
        > b_frag[WARP_N / MMA_N];


        // ------------------
        // A fragments
        // ------------------

        #pragma unroll
        for (int mi = 0;
             mi < WARP_M / MMA_M;
             ++mi) {

            int smem_m =
                warp_m_id * WARP_M
                + mi * MMA_M;

            const half* ptrA =
                &As[smem_m][0];

            wmma::load_matrix_sync(
                a_frag[mi],
                ptrA,
                CTA_K);
        }


        // ------------------
        // B fragments
        // ------------------

        #pragma unroll
        for (int ni = 0;
             ni < WARP_N / MMA_N;
             ++ni) {

            int smem_n =
                warp_n_id * WARP_N
                + ni * MMA_N;

            const half* ptrB =
                &Bs[0][smem_n];

            wmma::load_matrix_sync(
                b_frag[ni],
                ptrB,
                CTA_N);
        }


        // ====================================================
        // Warp-level MMA
        //
        // Each warp does:
        //
        // A0 x B0 -> C00
        // A0 x B1 -> C01
        // A1 x B0 -> C10
        // A1 x B1 -> C11
        // ====================================================

        #pragma unroll
        for (int mi = 0;
             mi < WARP_M / MMA_M;
             ++mi) {

            #pragma unroll
            for (int ni = 0;
                 ni < WARP_N / MMA_N;
                 ++ni) {

                wmma::mma_sync(
                    c_frag[mi][ni],
                    a_frag[mi],
                    b_frag[ni],
                    c_frag[mi][ni]);
            }
        }


        __syncthreads();
    }


    // ========================================================
    // Store accumulator fragments
    //
    // Register fragments -> Global Memory
    // ========================================================

    #pragma unroll
    for (int mi = 0;
         mi < WARP_M / MMA_M;
         ++mi) {

        #pragma unroll
        for (int ni = 0;
             ni < WARP_N / MMA_N;
             ++ni) {

            int global_m =
                cta_m
                + warp_m_id * WARP_M
                + mi * MMA_M;

            int global_n =
                cta_n
                + warp_n_id * WARP_N
                + ni * MMA_N;

            // Demo assumes dimensions are multiples
            // of CTA tile dimensions.
            float* ptrC =
                C + global_m * N + global_n;

            wmma::store_matrix_sync(
                ptrC,
                c_frag[mi][ni],
                N,
                wmma::mem_row_major);
        }
    }
}


// ============================================================
// Benchmark helper
// ============================================================

template <typename Launch>
float benchmark(Launch launch, int warmup, int repeat)
{
    for (int i = 0; i < warmup; ++i) {
        launch();
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        launch();
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;

    CHECK_CUDA(
        cudaEventElapsedTime(
            &ms,
            start,
            stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / repeat;
}


// ============================================================
// Main
// ============================================================

int main()
{
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    static_assert(M % CTA_M == 0);
    static_assert(N % CTA_N == 0);
    static_assert(K % CTA_K == 0);


    size_t sizeA =
        static_cast<size_t>(M) * K;

    size_t sizeB =
        static_cast<size_t>(K) * N;

    size_t sizeC =
        static_cast<size_t>(M) * N;


    std::vector<half> hA(sizeA);
    std::vector<half> hB(sizeB);

    std::vector<float> hC_base(sizeC);
    std::vector<float> hC_wmma(sizeC);


    // --------------------------------------------------------
    // Deterministic input
    // --------------------------------------------------------

    for (size_t i = 0; i < sizeA; ++i) {
        float x =
            static_cast<float>(
                static_cast<int>(i % 17) - 8)
            / 16.0f;

        hA[i] = __float2half(x);
    }

    for (size_t i = 0; i < sizeB; ++i) {
        float x =
            static_cast<float>(
                static_cast<int>(i % 13) - 6)
            / 16.0f;

        hB[i] = __float2half(x);
    }


    half* dA = nullptr;
    half* dB = nullptr;

    float* dC_base = nullptr;
    float* dC_wmma = nullptr;

    CHECK_CUDA(
        cudaMalloc(&dA,
                   sizeA * sizeof(half)));

    CHECK_CUDA(
        cudaMalloc(&dB,
                   sizeB * sizeof(half)));

    CHECK_CUDA(
        cudaMalloc(&dC_base,
                   sizeC * sizeof(float)));

    CHECK_CUDA(
        cudaMalloc(&dC_wmma,
                   sizeC * sizeof(float)));


    CHECK_CUDA(
        cudaMemcpy(
            dA,
            hA.data(),
            sizeA * sizeof(half),
            cudaMemcpyHostToDevice));

    CHECK_CUDA(
        cudaMemcpy(
            dB,
            hB.data(),
            sizeB * sizeof(half),
            cudaMemcpyHostToDevice));


    // ========================================================
    // Ordinary CUDA kernel
    // ========================================================

    dim3 base_block(
        BASE_TILE,
        BASE_TILE);

    dim3 base_grid(
        (N + BASE_TILE - 1) / BASE_TILE,
        (M + BASE_TILE - 1) / BASE_TILE);


    auto launch_base = [&]() {

        tiled_gemm<<<
            base_grid,
            base_block
        >>>(
            dA,
            dB,
            dC_base,
            M,
            N,
            K);
    };


    // ========================================================
    // Tensor Core WMMA kernel
    // ========================================================

    dim3 wmma_block(
        WMMA_THREADS);

    dim3 wmma_grid(
        N / CTA_N,
        M / CTA_M);


    auto launch_wmma = [&]() {

        wmma_gemm<<<
            wmma_grid,
            wmma_block
        >>>(
            dA,
            dB,
            dC_wmma,
            M,
            N,
            K);
    };


    float base_ms =
        benchmark(
            launch_base,
            5,
            20);

    CHECK_CUDA(cudaGetLastError());


    float wmma_ms =
        benchmark(
            launch_wmma,
            5,
            20);

    CHECK_CUDA(cudaGetLastError());


    CHECK_CUDA(
        cudaMemcpy(
            hC_base.data(),
            dC_base,
            sizeC * sizeof(float),
            cudaMemcpyDeviceToHost));

    CHECK_CUDA(
        cudaMemcpy(
            hC_wmma.data(),
            dC_wmma,
            sizeC * sizeof(float),
            cudaMemcpyDeviceToHost));


    // ========================================================
    // Correctness
    // ========================================================

    double max_abs_error = 0.0;

    for (size_t i = 0; i < sizeC; ++i) {

        double err =
            std::abs(
                static_cast<double>(hC_base[i]) -
                static_cast<double>(hC_wmma[i]));

        max_abs_error =
            std::max(
                max_abs_error,
                err);
    }


    // ========================================================
    // Performance
    // ========================================================

    double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    double base_tflops =
        flops /
        (base_ms * 1e9);

    double wmma_tflops =
        flops /
        (wmma_ms * 1e9);


    std::cout
        << "M=N=K=1024\n\n";

    std::cout
        << "CUDA tiled GEMM:\n"
        << "  time   = "
        << base_ms
        << " ms\n"
        << "  TFLOPS = "
        << base_tflops
        << "\n\n";

    std::cout
        << "WMMA Tensor Core GEMM:\n"
        << "  time   = "
        << wmma_ms
        << " ms\n"
        << "  TFLOPS = "
        << wmma_tflops
        << "\n\n";

    std::cout
        << "Speedup: "
        << base_ms / wmma_ms
        << "x\n";

    std::cout
        << "Max abs error: "
        << max_abs_error
        << "\n";


    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC_base);
    cudaFree(dC_wmma);

    return 0;
}