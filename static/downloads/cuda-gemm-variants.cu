#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)            \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";         \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ============================================================================
// Parameterized GEMM
//
//   Block tile : BM x BN x BK
//   Thread tile: TM x TN
//
// This teaching kernel always uses:
//   - shared-memory tiling
//   - register tiling
//
// The block-level schedule is generic.
// The hot thread-level micro-kernel is explicitly specialized for 1x4 / 2x4
// so that the compiler sees scalar accumulators directly.
// ============================================================================

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_tiled(const float* A,
                             const float* B,
                             float* C,
                             int N) {
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");

    constexpr int THREAD_ROWS = BM / TM;
    constexpr int THREAD_COLS = BN / TN;
    constexpr int THREADS = THREAD_ROWS * THREAD_COLS;

    static_assert(THREADS <= 1024, "Too many threads per block");
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of 32");

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int col_groups = BN / TN;

    const int row_group = tid / col_groups;
    const int col_group = tid % col_groups;
    const int row_in_tile = row_group * TM;
    const int col_in_tile = col_group * TN;

    const int row = blockIdx.y * BM + row_in_tile;
    const int col = blockIdx.x * BN + col_in_tile;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // ------------------------------------------------------------------------
    // 1 x 4 thread tile
    // ------------------------------------------------------------------------
    if constexpr (TM == 1 && TN == 4) {
        float c0 = 0.0f;
        float c1 = 0.0f;
        float c2 = 0.0f;
        float c3 = 0.0f;

        for (int tile = 0; tile < (N + BK - 1) / BK; ++tile) {
            // Cooperative load: A[BM, BK] -> As
#pragma unroll 1
            for (int idx = tid; idx < BM * BK; idx += blockDim.x * blockDim.y) {
                const int r = idx / BK;
                const int k = idx % BK;

                const int gr = blockIdx.y * BM + r;
                const int gk = tile * BK + k;

                As[r][k] =
                    (gr < N && gk < N) ? A[gr * N + gk] : 0.0f;
            }

            // Cooperative load: B[BK, BN] -> Bs
#pragma unroll 1
            for (int idx = tid; idx < BK * BN; idx += blockDim.x * blockDim.y) {
                const int k = idx / BN;
                const int c = idx % BN;

                const int gk = tile * BK + k;
                const int gc = blockIdx.x * BN + c;

                Bs[k][c] =
                    (gk < N && gc < N) ? B[gk * N + gc] : 0.0f;
            }

            __syncthreads();

#pragma unroll
            for (int k = 0; k < BK; ++k) {
                const float a = As[row_in_tile][k];

                const float b0 = Bs[k][col_in_tile + 0];
                const float b1 = Bs[k][col_in_tile + 1];
                const float b2 = Bs[k][col_in_tile + 2];
                const float b3 = Bs[k][col_in_tile + 3];

                c0 += a * b0;
                c1 += a * b1;
                c2 += a * b2;
                c3 += a * b3;
            }

            __syncthreads();
        }

        if (row < N) {
            if (col + 0 < N) C[row * N + col + 0] = c0;
            if (col + 1 < N) C[row * N + col + 1] = c1;
            if (col + 2 < N) C[row * N + col + 2] = c2;
            if (col + 3 < N) C[row * N + col + 3] = c3;
        }
    }

    // ------------------------------------------------------------------------
    // 2 x 4 thread tile
    // ------------------------------------------------------------------------
    else if constexpr (TM == 2 && TN == 4) {
        float c00 = 0.0f;
        float c01 = 0.0f;
        float c02 = 0.0f;
        float c03 = 0.0f;

        float c10 = 0.0f;
        float c11 = 0.0f;
        float c12 = 0.0f;
        float c13 = 0.0f;

        for (int tile = 0; tile < (N + BK - 1) / BK; ++tile) {
            // Cooperative load: A[BM, BK] -> As
#pragma unroll 1
            for (int idx = tid; idx < BM * BK; idx += blockDim.x * blockDim.y) {
                const int r = idx / BK;
                const int k = idx % BK;

                const int gr = blockIdx.y * BM + r;
                const int gk = tile * BK + k;

                As[r][k] =
                    (gr < N && gk < N) ? A[gr * N + gk] : 0.0f;
            }

            // Cooperative load: B[BK, BN] -> Bs
#pragma unroll 1
            for (int idx = tid; idx < BK * BN; idx += blockDim.x * blockDim.y) {
                const int k = idx / BN;
                const int c = idx % BN;

                const int gk = tile * BK + k;
                const int gc = blockIdx.x * BN + c;

                Bs[k][c] =
                    (gk < N && gc < N) ? B[gk * N + gc] : 0.0f;
            }

            __syncthreads();

#pragma unroll
            for (int k = 0; k < BK; ++k) {
                const float a0 = As[row_in_tile + 0][k];
                const float a1 = As[row_in_tile + 1][k];

                const float b0 = Bs[k][col_in_tile + 0];
                const float b1 = Bs[k][col_in_tile + 1];
                const float b2 = Bs[k][col_in_tile + 2];
                const float b3 = Bs[k][col_in_tile + 3];

                c00 += a0 * b0;
                c01 += a0 * b1;
                c02 += a0 * b2;
                c03 += a0 * b3;

                c10 += a1 * b0;
                c11 += a1 * b1;
                c12 += a1 * b2;
                c13 += a1 * b3;
            }

            __syncthreads();
        }

        if (row < N) {
            if (col + 0 < N) C[row * N + col + 0] = c00;
            if (col + 1 < N) C[row * N + col + 1] = c01;
            if (col + 2 < N) C[row * N + col + 2] = c02;
            if (col + 3 < N) C[row * N + col + 3] = c03;
        }

        if (row + 1 < N) {
            if (col + 0 < N) C[(row + 1) * N + col + 0] = c10;
            if (col + 1 < N) C[(row + 1) * N + col + 1] = c11;
            if (col + 2 < N) C[(row + 1) * N + col + 2] = c12;
            if (col + 3 < N) C[(row + 1) * N + col + 3] = c13;
        }
    }


    // ------------------------------------------------------------------------
    // 4 x 4 thread tile
    // Same block tile as 2x4: BM=32, BN=64, BK=16.
    // Fewer threads per block (128), but each thread computes twice as many C
    // elements and reuses each shared-memory operand more aggressively.
    // ------------------------------------------------------------------------
    else if constexpr (TM == 4 && TN == 4) {
        float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
        float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
        float c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
        float c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

        for (int tile = 0; tile < (N + BK - 1) / BK; ++tile) {
            // Cooperative load: A[BM, BK] -> As
#pragma unroll 1
            for (int idx = tid; idx < BM * BK; idx += blockDim.x * blockDim.y) {
                const int r = idx / BK;
                const int k = idx % BK;

                const int gr = blockIdx.y * BM + r;
                const int gk = tile * BK + k;

                As[r][k] =
                    (gr < N && gk < N) ? A[gr * N + gk] : 0.0f;
            }

            // Cooperative load: B[BK, BN] -> Bs
#pragma unroll 1
            for (int idx = tid; idx < BK * BN; idx += blockDim.x * blockDim.y) {
                const int k = idx / BN;
                const int c = idx % BN;

                const int gk = tile * BK + k;
                const int gc = blockIdx.x * BN + c;

                Bs[k][c] =
                    (gk < N && gc < N) ? B[gk * N + gc] : 0.0f;
            }

            __syncthreads();

#pragma unroll
            for (int k = 0; k < BK; ++k) {
                const float a0 = As[row_in_tile + 0][k];
                const float a1 = As[row_in_tile + 1][k];
                const float a2 = As[row_in_tile + 2][k];
                const float a3 = As[row_in_tile + 3][k];

                const float b0 = Bs[k][col_in_tile + 0];
                const float b1 = Bs[k][col_in_tile + 1];
                const float b2 = Bs[k][col_in_tile + 2];
                const float b3 = Bs[k][col_in_tile + 3];

                c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
            }

            __syncthreads();
        }

        if (row + 0 < N) {
            if (col + 0 < N) C[(row + 0) * N + col + 0] = c00;
            if (col + 1 < N) C[(row + 0) * N + col + 1] = c01;
            if (col + 2 < N) C[(row + 0) * N + col + 2] = c02;
            if (col + 3 < N) C[(row + 0) * N + col + 3] = c03;
        }
        if (row + 1 < N) {
            if (col + 0 < N) C[(row + 1) * N + col + 0] = c10;
            if (col + 1 < N) C[(row + 1) * N + col + 1] = c11;
            if (col + 2 < N) C[(row + 1) * N + col + 2] = c12;
            if (col + 3 < N) C[(row + 1) * N + col + 3] = c13;
        }
        if (row + 2 < N) {
            if (col + 0 < N) C[(row + 2) * N + col + 0] = c20;
            if (col + 1 < N) C[(row + 2) * N + col + 1] = c21;
            if (col + 2 < N) C[(row + 2) * N + col + 2] = c22;
            if (col + 3 < N) C[(row + 2) * N + col + 3] = c23;
        }
        if (row + 3 < N) {
            if (col + 0 < N) C[(row + 3) * N + col + 0] = c30;
            if (col + 1 < N) C[(row + 3) * N + col + 1] = c31;
            if (col + 2 < N) C[(row + 3) * N + col + 2] = c32;
            if (col + 3 < N) C[(row + 3) * N + col + 3] = c33;
        }
    }

    // ------------------------------------------------------------------------
    // 4 x 8 thread tile
    // ------------------------------------------------------------------------
    else if constexpr (TM == 4 && TN == 8) {
        float acc[4][8] = {};

        for (int tile = 0; tile < (N + BK - 1) / BK; ++tile) {
#pragma unroll 1
            for (int idx = tid; idx < BM * BK; idx += blockDim.x * blockDim.y) {
                const int r = idx / BK;
                const int k = idx % BK;
                const int gr = blockIdx.y * BM + r;
                const int gk = tile * BK + k;
                As[r][k] = (gr < N && gk < N) ? A[gr * N + gk] : 0.0f;
            }

#pragma unroll 1
            for (int idx = tid; idx < BK * BN; idx += blockDim.x * blockDim.y) {
                const int k = idx / BN;
                const int c = idx % BN;
                const int gk = tile * BK + k;
                const int gc = blockIdx.x * BN + c;
                Bs[k][c] = (gk < N && gc < N) ? B[gk * N + gc] : 0.0f;
            }

            __syncthreads();

#pragma unroll
            for (int k = 0; k < BK; ++k) {
                float a[4];
                float b[8];

#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    a[i] = As[row_in_tile + i][k];
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    b[j] = Bs[k][col_in_tile + j];
                }

#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i][j] += a[i] * b[j];
                    }
                }
            }

            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = row + i;
            if (r >= N) continue;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int c = col + j;
                if (c < N) C[r * N + c] = acc[i][j];
            }
        }
    }

    // ------------------------------------------------------------------------
    // 8 x 4 thread tile
    // ------------------------------------------------------------------------
    else if constexpr (TM == 8 && TN == 4) {
        float acc[8][4] = {};

        for (int tile = 0; tile < (N + BK - 1) / BK; ++tile) {
#pragma unroll 1
            for (int idx = tid; idx < BM * BK; idx += blockDim.x * blockDim.y) {
                const int r = idx / BK;
                const int k = idx % BK;
                const int gr = blockIdx.y * BM + r;
                const int gk = tile * BK + k;
                As[r][k] = (gr < N && gk < N) ? A[gr * N + gk] : 0.0f;
            }

#pragma unroll 1
            for (int idx = tid; idx < BK * BN; idx += blockDim.x * blockDim.y) {
                const int k = idx / BN;
                const int c = idx % BN;
                const int gk = tile * BK + k;
                const int gc = blockIdx.x * BN + c;
                Bs[k][c] = (gk < N && gc < N) ? B[gk * N + gc] : 0.0f;
            }

            __syncthreads();

#pragma unroll
            for (int k = 0; k < BK; ++k) {
                float a[8];
                float b[4];

#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    a[i] = As[row_in_tile + i][k];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    b[j] = Bs[k][col_in_tile + j];
                }

#pragma unroll
                for (int i = 0; i < 8; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[i][j] += a[i] * b[j];
                    }
                }
            }

            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = row + i;
            if (r >= N) continue;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = col + j;
                if (c < N) C[r * N + c] = acc[i][j];
            }
        }
    }

    else {
        static_assert((TM == 1 && TN == 4) ||
                      (TM == 2 && TN == 4) ||
                      (TM == 4 && TN == 4) ||
                      (TM == 4 && TN == 8) ||
                      (TM == 8 && TN == 4),
                      "Supported thread tiles: 1x4, 2x4, 4x4, 4x8 and 8x4");
    }
}

// ============================================================================
// 4x4 + vectorized C store
//
// This kernel intentionally keeps the same:
//   BM=32, BN=64, BK=16, TM=4, TN=4
//   cooperative loads
//   shared-memory layout
//   4x4 micro-kernel
//
// The only intended difference from the scalar 4x4 kernel is the final
// C write-back: 4 scalar float stores per row -> one float4 store per row.
//
// Fast-path requirement:
//   N % 64 == 0
// so every block is full and every float4 destination is 16-byte aligned.
// ============================================================================
__global__ void matmul_tiled_4x4_vecstore(
    const float* A, const float* B, float* C, int N) {

    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int THREAD_COLS = BN / TN;  // 16
    constexpr int THREADS = (BM / TM) * (BN / TN); // 128

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_row = tid / THREAD_COLS;
    const int thread_col = tid % THREAD_COLS;

    const int row = blockIdx.y * BM + thread_row * TM;
    const int col = blockIdx.x * BN + thread_col * TN;

    const int row_in_tile = thread_row * TM;
    const int col_in_tile = thread_col * TN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;
    float c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
    float c30 = 0.0f, c31 = 0.0f, c32 = 0.0f, c33 = 0.0f;

    for (int tile = 0; tile < N / BK; ++tile) {
#pragma unroll 1
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int r = idx / BK;
            const int k = idx % BK;
            const int gr = blockIdx.y * BM + r;
            const int gk = tile * BK + k;
            As[r][k] = A[static_cast<std::size_t>(gr) * N + gk];
        }

#pragma unroll 1
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            const int k = idx / BN;
            const int c = idx % BN;
            const int gk = tile * BK + k;
            const int gc = blockIdx.x * BN + c;
            Bs[k][c] = B[static_cast<std::size_t>(gk) * N + gc];
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float a0 = As[row_in_tile + 0][k];
            const float a1 = As[row_in_tile + 1][k];
            const float a2 = As[row_in_tile + 2][k];
            const float a3 = As[row_in_tile + 3][k];

            const float b0 = Bs[k][col_in_tile + 0];
            const float b1 = Bs[k][col_in_tile + 1];
            const float b2 = Bs[k][col_in_tile + 2];
            const float b3 = Bs[k][col_in_tile + 3];

            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        }

        __syncthreads();
    }

    // One 128-bit store per output row.
    *reinterpret_cast<float4*>(C + static_cast<std::size_t>(row + 0) * N + col) =
        make_float4(c00, c01, c02, c03);
    *reinterpret_cast<float4*>(C + static_cast<std::size_t>(row + 1) * N + col) =
        make_float4(c10, c11, c12, c13);
    *reinterpret_cast<float4*>(C + static_cast<std::size_t>(row + 2) * N + col) =
        make_float4(c20, c21, c22, c23);
    *reinterpret_cast<float4*>(C + static_cast<std::size_t>(row + 3) * N + col) =
        make_float4(c30, c31, c32, c33);
}

// ============================================================================
// 4x8 + padded A shared-memory tile
//
// Same schedule as the existing 4x8 kernel:
//   BM=32, BN=64, BK=16, TM=4, TN=8
//   64 threads/block
//
// Only intentional change:
//   As[BM][BK] -> As[BM][BK + 1]
//
// This changes the row stride in shared memory from 16 floats to 17 floats,
// perturbing the bank mapping while leaving the arithmetic work unchanged.
// ============================================================================
__global__ void matmul_tiled_4x8_pad(
    const float* A, const float* B, float* C, int N) {

    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 8;

    constexpr int THREAD_ROWS = BM / TM;   // 8
    constexpr int THREAD_COLS = BN / TN;   // 8
    constexpr int THREADS = THREAD_ROWS * THREAD_COLS; // 64

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_row = tid / THREAD_COLS;
    const int thread_col = tid % THREAD_COLS;

    const int row = blockIdx.y * BM + thread_row * TM;
    const int col = blockIdx.x * BN + thread_col * TN;

    const int row_in_tile = thread_row * TM;
    const int col_in_tile = thread_col * TN;

    // Padding is the only deliberate layout change.
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN];

    float acc[TM][TN] = {};

    for (int tile = 0; tile < (N + BK - 1) / BK; ++tile) {
#pragma unroll 1
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int r = idx / BK;
            const int k = idx % BK;

            const int gr = blockIdx.y * BM + r;
            const int gk = tile * BK + k;

            As[r][k] =
                (gr < N && gk < N)
                    ? A[static_cast<std::size_t>(gr) * N + gk]
                    : 0.0f;
        }

#pragma unroll 1
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            const int k = idx / BN;
            const int c = idx % BN;

            const int gk = tile * BK + k;
            const int gc = blockIdx.x * BN + c;

            Bs[k][c] =
                (gk < N && gc < N)
                    ? B[static_cast<std::size_t>(gk) * N + gc]
                    : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];

#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[row_in_tile + i][k];
            }

#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[k][col_in_tile + j];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row + i;
        if (r >= N) continue;

#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = col + j;
            if (c < N) {
                C[static_cast<std::size_t>(r) * N + c] = acc[i][j];
            }
        }
    }
}


// ============================================================================
// 4x8 + padded A and remapped/padded B shared-memory tiles
//
// A:
//   As[32][16] -> As[32][17]
//
// B:
//   logical columns 0..31  -> physical 0..31
//   logical columns 32..63 -> physical 33..64
//
// i.e. physical_col = logical_col + (logical_col >> 5)
//
// This inserts one pad element between the two 32-float halves of each B row,
// breaking the c / (c + 32) bank alias pattern.
// ============================================================================
__global__ void matmul_tiled_4x8_padAB(
    const float* A, const float* B, float* C, int N) {

    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 8;

    constexpr int THREAD_ROWS = BM / TM;
    constexpr int THREAD_COLS = BN / TN;
    constexpr int THREADS = THREAD_ROWS * THREAD_COLS;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_row = tid / THREAD_COLS;
    const int thread_col = tid % THREAD_COLS;

    const int row = blockIdx.y * BM + thread_row * TM;
    const int col = blockIdx.x * BN + thread_col * TN;

    const int row_in_tile = thread_row * TM;
    const int col_in_tile = thread_col * TN;

    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    float acc[TM][TN] = {};

    for (int tile = 0; tile < (N + BK - 1) / BK; ++tile) {
#pragma unroll 1
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int r = idx / BK;
            const int k = idx % BK;

            const int gr = blockIdx.y * BM + r;
            const int gk = tile * BK + k;

            As[r][k] =
                (gr < N && gk < N)
                    ? A[static_cast<std::size_t>(gr) * N + gk]
                    : 0.0f;
        }

#pragma unroll 1
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            const int k = idx / BN;
            const int c = idx % BN;

            const int gk = tile * BK + k;
            const int gc = blockIdx.x * BN + c;

            const int pc = c + (c >> 5);

            Bs[k][pc] =
                (gk < N && gc < N)
                    ? B[static_cast<std::size_t>(gk) * N + gc]
                    : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];

#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[row_in_tile + i][k];
            }

#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int logical_c = col_in_tile + j;
                const int pc = logical_c + (logical_c >> 5);
                b[j] = Bs[k][pc];
            }

#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row + i;
        if (r >= N) continue;

#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = col + j;
            if (c < N) {
                C[static_cast<std::size_t>(r) * N + c] = acc[i][j];
            }
        }
    }
}


// ============================================================================
// Launch / benchmark helpers
// ============================================================================

template <int BM, int BN, int BK, int TM, int TN>
void launch_matmul(const float* A,
                   const float* B,
                   float* C,
                   int N) {
    constexpr int THREADS = (BM / TM) * (BN / TN);

    // Keep the same physical CUDA block organization as the earlier kernels.
    const dim3 block(32, THREADS / 32);
    const dim3 grid(
        (N + BN - 1) / BN,
        (N + BM - 1) / BM
    );

    matmul_tiled<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, N);
}

template <int BM, int BN, int BK, int TM, int TN>
double benchmark(const float* A,
                 const float* B,
                 float* C,
                 int N,
                 int repeat) {
    constexpr int WARMUP = 100;
    constexpr int ROUNDS = 7;

    // Warm the GPU up long enough to reduce clock / power-state noise.
    for (int i = 0; i < WARMUP; ++i) {
        launch_matmul<BM, BN, BK, TM, TN>(A, B, C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> samples;
    samples.reserve(ROUNDS);

    for (int r = 0; r < ROUNDS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i) {
            launch_matmul<BM, BN, BK, TM, TN>(A, B, C, N);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        samples.push_back(static_cast<double>(total_ms) / repeat);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void launch_matmul_4x4_vecstore(
    const float* A, const float* B, float* C, int N) {

    const dim3 block(32, 4);  // 128 threads
    const dim3 grid(N / 64, N / 32);

    matmul_tiled_4x4_vecstore<<<grid, block>>>(A, B, C, N);
}

double benchmark_4x4_vecstore(
    const float* A,
    const float* B,
    float* C,
    int N,
    int repeat) {

    constexpr int WARMUP = 100;
    constexpr int ROUNDS = 7;

    for (int i = 0; i < WARMUP; ++i) {
        launch_matmul_4x4_vecstore(A, B, C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> samples;
    samples.reserve(ROUNDS);

    for (int r = 0; r < ROUNDS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i) {
            launch_matmul_4x4_vecstore(A, B, C, N);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        samples.push_back(static_cast<double>(total_ms) / repeat);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}


void launch_matmul_4x8_pad(
    const float* A, const float* B, float* C, int N) {

    const dim3 block(32, 2);  // 64 threads
    const dim3 grid(
        (N + 64 - 1) / 64,
        (N + 32 - 1) / 32);

    matmul_tiled_4x8_pad<<<grid, block>>>(A, B, C, N);
}

double benchmark_4x8_pad(
    const float* A,
    const float* B,
    float* C,
    int N,
    int repeat) {

    constexpr int WARMUP = 100;
    constexpr int ROUNDS = 7;

    for (int i = 0; i < WARMUP; ++i) {
        launch_matmul_4x8_pad(A, B, C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> samples;
    samples.reserve(ROUNDS);

    for (int r = 0; r < ROUNDS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i) {
            launch_matmul_4x8_pad(A, B, C, N);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        samples.push_back(static_cast<double>(total_ms) / repeat);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}


void launch_matmul_4x8_padAB(
    const float* A, const float* B, float* C, int N) {

    const dim3 block(32, 2);
    const dim3 grid(
        (N + 64 - 1) / 64,
        (N + 32 - 1) / 32);

    matmul_tiled_4x8_padAB<<<grid, block>>>(A, B, C, N);
}

double benchmark_4x8_padAB(
    const float* A,
    const float* B,
    float* C,
    int N,
    int repeat) {

    constexpr int WARMUP = 100;
    constexpr int ROUNDS = 7;

    for (int i = 0; i < WARMUP; ++i) {
        launch_matmul_4x8_padAB(A, B, C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> samples;
    samples.reserve(ROUNDS);

    for (int r = 0; r < ROUNDS; ++r) {
        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < repeat; ++i) {
            launch_matmul_4x8_padAB(A, B, C, N);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        samples.push_back(static_cast<double>(total_ms) / repeat);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}


bool spot_check(const std::vector<float>& A,
                const std::vector<float>& B,
                const std::vector<float>& C,
                int N) {
    const int points[][2] = {
        {0, 0},
        {0, N - 1},
        {N - 1, 0},
        {N - 1, N - 1},
        {N / 2, N / 2},
        {N / 3, N / 5},
    };

    for (const auto& p : points) {
        const int row = p[0];
        const int col = p[1];

        float expected = 0.0f;
        for (int k = 0; k < N; ++k) {
            expected += A[static_cast<std::size_t>(row) * N + k] *
                        B[static_cast<std::size_t>(k) * N + col];
        }

        const float actual = C[static_cast<std::size_t>(row) * N + col];
        const float error = std::fabs(actual - expected);
        const float tolerance = 1e-3f + 1e-3f * std::fabs(expected);

        if (error > tolerance) {
            std::cerr << "check failed at (" << row << ", " << col << ")"
                      << ": expected=" << expected
                      << ", actual=" << actual
                      << ", error=" << error << '\n';
            return false;
        }
    }

    return true;
}

template <int BM, int BN, int BK, int TM, int TN>
void run_config(const char* name,
                const std::vector<float>& h_A,
                const std::vector<float>& h_B,
                std::vector<float>& h_C,
                const float* d_A,
                const float* d_B,
                float* d_C,
                std::size_t bytes,
                int N,
                int repeat) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr int SHARED_BYTES =
        (BM * BK + BK * BN) * static_cast<int>(sizeof(float));

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    const double ms =
        benchmark<BM, BN, BK, TM, TN>(d_A, d_B, d_C, N, repeat);

    const double gflops =
        2.0 * static_cast<double>(N) * N * N / (ms * 1.0e6);

    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    const bool ok = spot_check(h_A, h_B, h_C, N);

    std::cout
        << name << '\n'
        << "  BM=" << BM
        << ", BN=" << BN
        << ", BK=" << BK
        << ", TM=" << TM
        << ", TN=" << TN << '\n'
        << "  threads/block=" << THREADS
        << ", shared/block=" << SHARED_BYTES / 1024.0 << " KiB\n"
        << "  time=" << ms << " ms"
        << ", performance=" << gflops << " GFLOPS"
        << ", check=" << (ok ? "OK" : "FAIL")
        << "\n\n";
}

void run_4x4_vecstore(
    const std::vector<float>& h_A,
    const std::vector<float>& h_B,
    std::vector<float>& h_C,
    const float* d_A,
    const float* d_B,
    float* d_C,
    std::size_t bytes,
    int N,
    int repeat) {

    if (N % 64 != 0) {
        std::cerr
            << "4x4vec requires N to be divisible by 64 "
            << "(clean full-tile/vector-store experiment)\n";
        return;
    }

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    const double ms =
        benchmark_4x4_vecstore(d_A, d_B, d_C, N, repeat);

    const double gflops =
        2.0 * static_cast<double>(N) * N * N / (ms * 1.0e6);

    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    const bool ok = spot_check(h_A, h_B, h_C, N);

    std::cout
        << "tiled_32x64x16_thread_4x4_vecstore\n"
        << "  BM=32, BN=64, BK=16, TM=4, TN=4\n"
        << "  threads/block=128, shared/block=6 KiB\n"
        << "  C store=float4 (16 B/thread/row)\n"
        << "  time=" << ms << " ms"
        << ", performance=" << gflops << " GFLOPS"
        << ", check=" << (ok ? "OK" : "FAIL")
        << "\n\n";
}


void run_4x8_pad(
    const std::vector<float>& h_A,
    const std::vector<float>& h_B,
    std::vector<float>& h_C,
    const float* d_A,
    const float* d_B,
    float* d_C,
    std::size_t bytes,
    int N,
    int repeat) {

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    const double ms =
        benchmark_4x8_pad(d_A, d_B, d_C, N, repeat);

    const double gflops =
        2.0 * static_cast<double>(N) * N * N / (ms * 1.0e6);

    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    const bool ok = spot_check(h_A, h_B, h_C, N);

    constexpr int PADDED_SHARED_BYTES =
        (32 * (16 + 1) + 16 * 64) * static_cast<int>(sizeof(float));

    std::cout
        << "tiled_32x64x16_thread_4x8_padA\n"
        << "  BM=32, BN=64, BK=16, TM=4, TN=8\n"
        << "  threads/block=64, shared/block="
        << PADDED_SHARED_BYTES / 1024.0 << " KiB\n"
        << "  As layout=[32][17] (BK+1 padding)\n"
        << "  time=" << ms << " ms"
        << ", performance=" << gflops << " GFLOPS"
        << ", check=" << (ok ? "OK" : "FAIL")
        << "\n\n";
}


void run_4x8_padAB(
    const std::vector<float>& h_A,
    const std::vector<float>& h_B,
    std::vector<float>& h_C,
    const float* d_A,
    const float* d_B,
    float* d_C,
    std::size_t bytes,
    int N,
    int repeat) {

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    const double ms =
        benchmark_4x8_padAB(d_A, d_B, d_C, N, repeat);

    const double gflops =
        2.0 * static_cast<double>(N) * N * N / (ms * 1.0e6);

    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    const bool ok = spot_check(h_A, h_B, h_C, N);

    constexpr int SHARED_BYTES =
        (32 * (16 + 1) + 16 * (64 + 1)) * static_cast<int>(sizeof(float));

    std::cout
        << "tiled_32x64x16_thread_4x8_padAB\n"
        << "  BM=32, BN=64, BK=16, TM=4, TN=8\n"
        << "  threads/block=64, shared/block="
        << SHARED_BYTES / 1024.0 << " KiB\n"
        << "  As layout=[32][17]\n"
        << "  Bs layout=[16][65], logical c -> c + (c >> 5)\n"
        << "  time=" << ms << " ms"
        << ", performance=" << gflops << " GFLOPS"
        << ", check=" << (ok ? "OK" : "FAIL")
        << "\n\n";
}


int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 2048;
    const int repeat = argc > 2 ? std::atoi(argv[2]) : 10;
    const std::string config = argc > 3 ? argv[3] : "2x4";

    if (N <= 0 || repeat <= 0) {
        std::cerr << "usage: ./matmul [N] [repeat] [1x4|2x4|4x4|4x4vec|4x8|4x8pad|8x4|all]\n";
        return 1;
    }

    const std::size_t elements = static_cast<std::size_t>(N) * N;
    const std::size_t bytes = elements * sizeof(float);

    std::vector<float> h_A(elements);
    std::vector<float> h_B(elements);
    std::vector<float> h_C(elements);

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            const std::size_t idx =
                static_cast<std::size_t>(row) * N + col;

            h_A[idx] =
                static_cast<float>((row * 17 + col * 13) % 101 - 50) / 50.0f;
            h_B[idx] =
                static_cast<float>((row * 19 + col * 23) % 97 - 48) / 48.0f;
        }
    }

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    std::cout << "N=" << N
              << ", repeat=" << repeat
              << ", config=" << config << "\n\n";

    if (config == "1x4" || config == "all") {
        run_config<32, 32, 16, 1, 4>(
            "tiled_32x32x16_thread_1x4",
            h_A, h_B, h_C,
            d_A, d_B, d_C,
            bytes, N, repeat);
    }

    if (config == "2x4" || config == "all") {
        run_config<32, 64, 16, 2, 4>(
            "tiled_32x64x16_thread_2x4",
            h_A, h_B, h_C,
            d_A, d_B, d_C,
            bytes, N, repeat);
    }

    if (config == "4x4" || config == "all") {
        run_config<32, 64, 16, 4, 4>(
            "tiled_32x64x16_thread_4x4",
            h_A, h_B, h_C,
            d_A, d_B, d_C,
            bytes, N, repeat);
    }

    if (config == "4x4vec" || config == "all") {
        run_4x4_vecstore(
            h_A, h_B, h_C,
            d_A, d_B, d_C,
            bytes, N, repeat);
    }

    if (config == "4x8" || config == "all") {
        run_config<32, 64, 16, 4, 8>(
            "tiled_32x64x16_thread_4x8",
            h_A, h_B, h_C,
            d_A, d_B, d_C,
            bytes, N, repeat);
    }

    if (config == "4x8pad" || config == "all") {
        run_4x8_pad(
            h_A, h_B, h_C,
            d_A, d_B, d_C,
            bytes, N, repeat);
    }

    if (config == "4x8padAB" || config == "all") {
        run_4x8_padAB(
            h_A, h_B, h_C,
            d_A, d_B, d_C,
            bytes, N, repeat);
    }

    if (config == "8x4" || config == "all") {
        run_config<32, 64, 16, 8, 4>(
            "tiled_32x64x16_thread_8x4",
            h_A, h_B, h_C,
            d_A, d_B, d_C,
            bytes, N, repeat);
    }

    if (config != "1x4" && config != "2x4" &&
        config != "4x4" && config != "4x4vec" && config != "4x8" && config != "4x8pad" && config != "4x8padAB" &&
        config != "8x4" && config != "all") {
        std::cerr << "unknown config: " << config
                  << "\nvalid configs: 1x4 | 2x4 | 4x4 | 4x4vec | 4x8 | 4x8pad | 4x8padAB | 8x4 | all\n";
        return 1;
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
