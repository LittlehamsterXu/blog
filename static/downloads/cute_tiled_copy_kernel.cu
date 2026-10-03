#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

template <class TiledCopy>
__global__ void copy_kernel(
    float const* src,
    float* dst,
    TiledCopy tiled_copy)
{
    // 32 x 8, left-major:
    //
    // offset = m + 32*n
    //
    auto layout =
        make_layout(
            make_shape(Int<32>{}, Int<8>{})
        );

    auto S =
        make_tensor(
            make_gmem_ptr(src),
            layout
        );

    auto D =
        make_tensor(
            make_gmem_ptr(dst),
            layout
        );

    // Current thread's copy mapping
    auto thr_copy =
        tiled_copy.get_thread_slice(threadIdx.x);

    // Thread-local views
    auto tS = thr_copy.partition_S(S);
    auto tD = thr_copy.partition_D(D);

    // Register-backed fragment
    auto frag =
        make_fragment_like(tD);

    if (threadIdx.x == 0) {
        printf("tS shape: ");
        print(shape(tS));
        printf("\n");

        printf("frag shape: ");
        print(shape(frag));
        printf("\n");
    }

    // GMEM -> registers
    copy(tiled_copy, tS, frag);

    // registers -> GMEM
    copy(tiled_copy, frag, tD);
}

int main()
{
    constexpr int M = 32;
    constexpr int N = 8;
    constexpr int Numel = M * N;

    float* src;
    float* dst;

    cudaMallocManaged(&src, Numel * sizeof(float));
    cudaMallocManaged(&dst, Numel * sizeof(float));

    for (int i = 0; i < Numel; ++i) {
        src[i] = float(i);
        dst[i] = -1.0f;
    }

    auto tiled_copy =
        make_tiled_copy(
            Copy_Atom<
                UniversalCopy<uint128_t>,
                float
            >{},
            Layout<Shape<_8,_8>>{},
            Layout<Shape<_4,_1>>{}
        );

    copy_kernel<<<1, 64>>>(
        src,
        dst,
        tiled_copy
    );

    cudaDeviceSynchronize();

    bool ok = true;

    for (int i = 0; i < Numel; ++i) {
        if (src[i] != dst[i]) {
            printf(
                "Mismatch at %d: src=%f dst=%f\n",
                i,
                src[i],
                dst[i]
            );
            ok = false;
            break;
        }
    }

    printf(
        "Verification: %s\n",
        ok ? "PASS" : "FAIL"
    );

    cudaFree(src);
    cudaFree(dst);

    return 0;
}