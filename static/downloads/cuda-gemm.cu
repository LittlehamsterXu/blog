#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

// CUDA_CHECK 宏用于检查 CUDA API 调用的返回值，如果调用失败，则打印错误信息并退出程序。
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << ", code: " << err << ", reason: " \
                      << cudaGetErrorString(err) << '\n'; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// V1 naive 实现：一个线程负责计算结果矩阵 C 中的一个元素。
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    // 根据 block 和 thread 的编号，找到当前线程负责的行、列。
    // cuda 习惯 x → column , y → row
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 矩阵大小不一定正好是 block 大小的整数倍，因此需要检查边界。
    if (row < M && col < N) {
        float sum = 0.0f;

        // 当前线程计算 C[row][col]。
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// V2 shared memory 实现：每个 block 负责计算 C 中一个 tile，使用共享内存缓存 A 和 B 的 tile。
constexpr int TILE = 16;

__global__ void matmul_shared(const float* A, const float* B, float* C, int M, int K, int N) {
    
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    // 每个 block 都有自己的一份 shared memory。
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    float sum = 0.0f;

    // 沿 K 维一块一块处理。
    for (int tile = 0; tile < (K + TILE - 1) / TILE; ++tile) {

        // 当前线程负责从 A 搬一个元素。
        const int a_col = tile * TILE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 当前线程负责从 B 搬一个元素。
        const int b_row = tile * TILE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 等整个 block 把 A/B tile 搬完。
        __syncthreads();

        // 现在所有数据都在 shared memory 中。
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // 确保所有线程用完当前 tile，
        // 才能覆盖 shared memory 加载下一块。
        __syncthreads();
    }

    // 只有有效线程负责把结果写回 C。
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 32 x 8 block：一个 warp 正好覆盖输出矩阵的一整行。
constexpr int BLOCK_X_32X8 = 32;
constexpr int BLOCK_Y_32X8 = 8;
constexpr int TILE_K_32X8 = 32;

__global__ void matmul_shared_32x8(const float* A, const float* B, float* C, int M, int K, int N) {
    const int row = blockIdx.y * BLOCK_Y_32X8 + threadIdx.y;
    const int col = blockIdx.x * BLOCK_X_32X8 + threadIdx.x;
    __shared__ float As[BLOCK_Y_32X8][TILE_K_32X8];
    __shared__ float Bs[TILE_K_32X8][BLOCK_X_32X8];
    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_K_32X8 - 1) / TILE_K_32X8; ++tile) {
        const int a_col = tile * TILE_K_32X8 + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 8 行线程共同加载 32 行 B，每个线程加载 4 个元素。
        for (int b_row = threadIdx.y; b_row < TILE_K_32X8; b_row += BLOCK_Y_32X8) {
            const int global_b_row = tile * TILE_K_32X8 + b_row;
            if (global_b_row < K && col < N) {
                Bs[b_row][threadIdx.x] = B[global_b_row * N + col];
            } else {
                Bs[b_row][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();
        for (int k = 0; k < TILE_K_32X8; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 32 x 32 block：每个 block 有 1024 个线程，整个 tile 都由一个 block 计算。
constexpr int TILE_32 = 32;

__global__ void matmul_shared_32x32(const float* A, const float* B, float* C, int M, int K, int N) {
    const int row = blockIdx.y * TILE_32 + threadIdx.y;
    const int col = blockIdx.x * TILE_32 + threadIdx.x;
    __shared__ float As[TILE_32][TILE_32];
    __shared__ float Bs[TILE_32][TILE_32];
    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_32 - 1) / TILE_32; ++tile) {
        
        const int a_col = tile * TILE_32 + threadIdx.x;
        const int b_row = tile * TILE_32 + threadIdx.y;

        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        for (int k = 0; k < TILE_32; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// V3：32 x 8 个线程共同覆盖一个 32 x 32 的输出 tile。
// 每个线程计算同一行中连续的 4 个 C 元素，也就是 1 x 4 register tile。
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 16;
constexpr int TN = 4;
constexpr int COL_GROUPS = BN / TN;

__global__ void matmul_reg_1x4(const float* A, const float* B, float* C, int M, int K, int N) {
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int linear_tid = ty * blockDim.x + tx;

    // 32 x 8 个线程共有 256 个线程。
    // 每 8 个线程负责一行，每个线程负责连续 4 列，正好覆盖 32 x 32。
    const int row_in_tile = linear_tid / COL_GROUPS;
    const int col_in_tile = (linear_tid % COL_GROUPS) * TN;
    const int row = blockIdx.y * BM + row_in_tile;
    const int col = blockIdx.x * BN + col_in_tile;

    // K 方向每次处理 BK 个元素：A tile 是 32 x 16，B tile 是 16 x 32。
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 4 个累加变量会放在寄存器中，分别对应当前线程负责的 4 个输出。
    // 写成独立变量，初学时更容易看清每个结果的计算过程。
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
        // 合作加载 A tile。A tile 有 32 x 16 个元素，256 个线程各加载两次。
        for (int index = linear_tid; index < BM * BK; index += blockDim.x * blockDim.y) {
            const int a_row_in_tile = index / BK;
            const int a_col_in_tile = index % BK;
            const int a_row = blockIdx.y * BM + a_row_in_tile;
            const int a_col = tile * BK + a_col_in_tile;

            if (a_row < M && a_col < K) {
                As[a_row_in_tile][a_col_in_tile] = A[a_row * K + a_col];
            } else {
                As[a_row_in_tile][a_col_in_tile] = 0.0f;
            }
        }

        // 合作加载 B tile。B tile 有 16 x 32 个元素，256 个线程各加载两次。
        for (int index = linear_tid; index < BK * BN; index += blockDim.x * blockDim.y) {
            const int b_row_in_tile = index / BN;
            const int b_col_in_tile = index % BN;
            const int b_row = tile * BK + b_row_in_tile;
            const int b_col = blockIdx.x * BN + b_col_in_tile;

            if (b_row < K && b_col < N) {
                Bs[b_row_in_tile][b_col_in_tile] = B[b_row * N + b_col];
            } else {
                Bs[b_row_in_tile][b_col_in_tile] = 0.0f;
            }
        }

        // 等待整个 block 完成 tile 加载。
        __syncthreads();

        // 一个 A 值同时更新 4 个寄存器累加器。
        for (int k = 0; k < BK; ++k) {
            const float a = As[row_in_tile][k];
            sum0 += a * Bs[k][col_in_tile + 0];
            sum1 += a * Bs[k][col_in_tile + 1];
            sum2 += a * Bs[k][col_in_tile + 2];
            sum3 += a * Bs[k][col_in_tile + 3];
        }

        // 确保所有线程都用完当前 tile，再加载下一块。
        __syncthreads();
    }

    // 处理边界 tile，只写回有效列。
    if (row < M) {
        if (col + 0 < N) C[row * N + col + 0] = sum0;
        if (col + 1 < N) C[row * N + col + 1] = sum1;
        if (col + 2 < N) C[row * N + col + 2] = sum2;
        if (col + 3 < N) C[row * N + col + 3] = sum3;
    }
}

// V4：每个线程计算 2 行 x 4 列，一共 8 个输出。
// 继续使用 32 x 8 个线程，因此一个 block 覆盖 32 x 64 的输出 tile。
constexpr int BM_2X4 = 32;
constexpr int BN_2X4 = 64;
constexpr int BK_2X4 = 16;
constexpr int TM_2X4 = 2;
constexpr int TN_2X4 = 4;
constexpr int COL_GROUPS_2X4 = BN_2X4 / TN_2X4;

__global__ void matmul_reg_2x4(const float* A, const float* B, float* C, int M, int K, int N) {
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int linear_tid = ty * blockDim.x + tx;

    // 256 个线程映射成 16 个二行组 x 16 个四列组。
    const int row_group = linear_tid / COL_GROUPS_2X4;
    const int col_group = linear_tid % COL_GROUPS_2X4;
    const int row_in_tile = row_group * TM_2X4;
    const int col_in_tile = col_group * TN_2X4;
    const int row = blockIdx.y * BM_2X4 + row_in_tile;
    const int col = blockIdx.x * BN_2X4 + col_in_tile;

    // V4 暂时不做 padding，先观察 register tiling 本身的效果。
    __shared__ float As[BM_2X4][BK_2X4];
    __shared__ float Bs[BK_2X4][BN_2X4];

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;

    for (int tile = 0; tile < (K + BK_2X4 - 1) / BK_2X4; ++tile) {
        // 合作加载 A tile。
        for (int index = linear_tid; index < BM_2X4 * BK_2X4; index += blockDim.x * blockDim.y) {
            const int a_row_in_tile = index / BK_2X4;
            const int a_col_in_tile = index % BK_2X4;
            const int a_row = blockIdx.y * BM_2X4 + a_row_in_tile;
            const int a_col = tile * BK_2X4 + a_col_in_tile;

            if (a_row < M && a_col < K) {
                As[a_row_in_tile][a_col_in_tile] = A[a_row * K + a_col];
            } else {
                As[a_row_in_tile][a_col_in_tile] = 0.0f;
            }
        }

        // 合作加载 B tile。
        for (int index = linear_tid; index < BK_2X4 * BN_2X4; index += blockDim.x * blockDim.y) {
            const int b_row_in_tile = index / BN_2X4;
            const int b_col_in_tile = index % BN_2X4;
            const int b_row = tile * BK_2X4 + b_row_in_tile;
            const int b_col = blockIdx.x * BN_2X4 + b_col_in_tile;

            if (b_row < K && b_col < N) {
                Bs[b_row_in_tile][b_col_in_tile] = B[b_row * N + b_col];
            } else {
                Bs[b_row_in_tile][b_col_in_tile] = 0.0f;
            }
        }

        __syncthreads();

        // 每个 k 同时更新两行、四列的 8 个寄存器累加器。
        for (int k = 0; k < BK_2X4; ++k) {
            const float a0 = As[row_in_tile + 0][k];
            const float a1 = As[row_in_tile + 1][k];
            const float b0 = Bs[k][col_in_tile + 0];
            const float b1 = Bs[k][col_in_tile + 1];
            const float b2 = Bs[k][col_in_tile + 2];
            const float b3 = Bs[k][col_in_tile + 3];

            sum0 += a0 * b0;
            sum1 += a0 * b1;
            sum2 += a0 * b2;
            sum3 += a0 * b3;
            sum4 += a1 * b0;
            sum5 += a1 * b1;
            sum6 += a1 * b2;
            sum7 += a1 * b3;
        }

        __syncthreads();
    }

    if (row < M) {
        if (col + 0 < N) C[row * N + col + 0] = sum0;
        if (col + 1 < N) C[row * N + col + 1] = sum1;
        if (col + 2 < N) C[row * N + col + 2] = sum2;
        if (col + 3 < N) C[row * N + col + 3] = sum3;
    }
    if (row + 1 < M) {
        if (col + 0 < N) C[(row + 1) * N + col + 0] = sum4;
        if (col + 1 < N) C[(row + 1) * N + col + 1] = sum5;
        if (col + 2 < N) C[(row + 1) * N + col + 2] = sum6;
        if (col + 3 < N) C[(row + 1) * N + col + 3] = sum7;
    }
}



// 用 host wrapper 把不同 kernel 统一成同一种函数指针类型。
using KernelLauncher = void (*)(const float*, const float*, float*, int, dim3, dim3);

void launch_naive(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block) {
    matmul_naive<<<grid, block>>>(A, B, C, N, N, N);
}

void launch_shared(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block) {
    matmul_shared<<<grid, block>>>(A, B, C, N, N, N);
}

void launch_shared_32x8(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block) {
    matmul_shared_32x8<<<grid, block>>>(A, B, C, N, N, N);
}

void launch_shared_32x32(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block) {
    matmul_shared_32x32<<<grid, block>>>(A, B, C, N, N, N);
}

void launch_reg_1x4(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block) {
    matmul_reg_1x4<<<grid, block>>>(A, B, C, N, N, N);
}

void launch_reg_2x4(const float* A, const float* B, float* C, int N, dim3 grid, dim3 block) {
    matmul_reg_2x4<<<grid, block>>>(A, B, C, N, N, N);
}

void benchmark_kernel(const char* name, KernelLauncher launch, dim3 grid, dim3 block, const float* d_A, const float* d_B, float* d_C, int N, int repeat) {

    // 先预热一次，让 CUDA 完成上下文初始化。
    launch(d_A, d_B, d_C, N, grid, block);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        launch(d_A, d_B, d_C, N, grid, block);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    const double average_ms = milliseconds / repeat;
    const double operations = 2.0 * static_cast<double>(N) * N * N;
    const double gflops = operations / (average_ms * 1.0e6);

    std::cout << name << ": " << average_ms << " ms, " << gflops << " GFLOPS\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void run_test(const char* name, KernelLauncher launch, dim3 grid, dim3 block, const float* d_A, const float* d_B, float* d_C, std::vector<float>& h_C, const std::vector<float>& h_expected, std::size_t bytes, int N, int repeat) {
    benchmark_kernel(name, launch, grid, block, d_A, d_B, d_C, N, repeat);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    float max_error = 0.0f;
    for (std::size_t i = 0; i < h_C.size(); ++i) {
        const float error = std::fabs(h_C[i] - h_expected[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (error > 1e-3f) {
            correct = false;
        }
    }

    std::cout << name << " result: " << (correct ? "correct" : "wrong")
              << ", max error = " << max_error << '\n';
}

int main(int argc, char** argv) {
    // 默认测试 512 x 512 矩阵，也可以通过命令行修改。
    const int N = argc > 1 ? std::atoi(argv[1]) : 512;
    const int repeat = argc > 2 ? std::atoi(argv[2]) : 10;

    if (N <= 0 || repeat <= 0) {
        std::cerr << "N 和 repeat 必须大于 0\n";
        return 1;
    }

    const std::size_t elements =
        static_cast<std::size_t>(N) * N;
    const std::size_t bytes = elements * sizeof(float);

    // 使用有规律但不相同的数，方便暴露行列索引错误。
    std::vector<float> h_A(elements);
    std::vector<float> h_B(elements);
    std::vector<float> h_C(elements, 0.0f);
    std::vector<float> h_expected(elements, 0.0f);

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * N + col;
            h_A[index] = static_cast<float>((row * 17 + col * 13) % 101 - 50) / 50.0f;
            h_B[index] = static_cast<float>((row * 19 + col * 23) % 97 - 48) / 48.0f;
        }
    }

    // 在 CPU 上计算参考结果，之后逐元素比较 GPU 输出。
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += h_A[static_cast<std::size_t>(row) * N + k]
                    * h_B[static_cast<std::size_t>(k) * N + col];
            }
            h_expected[static_cast<std::size_t>(row) * N + col] = sum;
        }
    }

    // d_ 开头的指针指向 GPU 显存，h_ 开头的数组在 CPU 内存中。
    float* d_A;
    float* d_B;
    float* d_C;

    // 在 GPU 上申请内存。
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 把输入矩阵从 CPU 复制到 GPU。
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    std::cout << "N = " << N << ", repeat = " << repeat << '\n';
    const dim3 block_16x16(16, 16);
    const dim3 block_32x8(32, 8);
    const dim3 block_32x32(32, 32);
    const dim3 grid_16x16((N + 15) / 16, (N + 15) / 16);
    const dim3 grid_32x8((N + 31) / 32, (N + 7) / 8);
    const dim3 grid_32x32((N + 31) / 32, (N + 31) / 32);
    // V3 每个 block 覆盖 32 列、32 行，所以 grid 的两个方向都按 32 计算。
    const dim3 grid_reg_1x4((N + BN - 1) / BN, (N + BM - 1) / BM);
    // V4 每个 block 覆盖 64 列、32 行，所以 grid x 按 64、grid y 按 32 计算。
    const dim3 grid_reg_2x4((N + BN_2X4 - 1) / BN_2X4, (N + BM_2X4 - 1) / BM_2X4);
    run_test("naive_16x16", launch_naive, grid_16x16, block_16x16, d_A, d_B, d_C, h_C, h_expected, bytes, N, repeat);
    run_test("shared_16x16", launch_shared, grid_16x16, block_16x16, d_A, d_B, d_C, h_C, h_expected, bytes, N, repeat);
    run_test("shared_32x8", launch_shared_32x8, grid_32x8, block_32x8, d_A, d_B, d_C, h_C, h_expected, bytes, N, repeat);
    run_test("shared_32x32", launch_shared_32x32, grid_32x32, block_32x32, d_A, d_B, d_C, h_C, h_expected, bytes, N, repeat);
    run_test("reg_1x4_32x8", launch_reg_1x4, grid_reg_1x4, block_32x8, d_A, d_B, d_C, h_C, h_expected, bytes, N, repeat);
    run_test("reg_2x4_32x8", launch_reg_2x4, grid_reg_2x4, block_32x8, d_A, d_B, d_C, h_C, h_expected, bytes, N, repeat);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

}
