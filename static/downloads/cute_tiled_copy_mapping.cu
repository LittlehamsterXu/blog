#include <cstdio>

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

int main() {
    using Element = float;

    // ThreadLayout = 8 x 8
    // ValueLayout  = 4 x 1
    // CopyAtom     = 128-bit = 4 floats
    auto tiled_copy = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, Element>{},
        Layout<Shape<_8, _8>>{},
        Layout<Shape<_4, _1>>{}
    );

    // TiledCopy 自己覆盖的完整 tile shape
    using TiledCopy = decltype(tiled_copy);
    auto tiler_mn = typename TiledCopy::Tiler_MN{};

    auto tile_mn =
        product_each(
            shape(
                logical_divide(
                    make_layout(Shape<_1, _1>{}),
                    tiler_mn
                )
            )
        );

    printf("Tile shape: ");
    print(tile_mn);
    printf("\n");

    // identity tensor:
    // 每个位置保存的就是它自己的逻辑坐标 (m,n)
    auto ref = make_identity_tensor(tile_mn);

    // 得到真正的:
    //
    //   (thread_id, value_id) -> (m,n)
    //
    auto tv = tiled_copy.tidfrg_S(ref)(_, _, Int<0>{});

    printf("TV shape: ");
    print(shape(tv));
    printf("\n\n");

    int tids[] = {0, 1, 7, 8, 9};

    for (int tid : tids) {
        printf("Thread %d:\n", tid);

        for (int vid = 0; vid < size<1>(tv); ++vid) {
            auto [m, n] = tv(tid, vid);

            printf(
                "  V%d -> (%d, %d)\n",
                vid,
                int(m),
                int(n)
            );
        }

        printf("\n");
    }

    return 0;
}