#include <cstdio>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>

using namespace cute;

int main() {
    auto tiled_mma =
        make_tiled_mma(
            SM80_16x8x16_F16F16F16F16_TN{},
            Layout<Shape<_1,_1>>{},
            Tile<_16,_8,_16>{}
        );

    printf("TiledMMA:\n");
    print(tiled_mma);
    printf("\n\n");

    printf("size(tiled_mma) = ");
    print(size(tiled_mma));
    printf("\n");

    printf("tile shape = (");
    print(tile_size<0>(tiled_mma));
    printf(", ");
    print(tile_size<1>(tiled_mma));
    printf(", ");
    print(tile_size<2>(tiled_mma));
    printf(")\n");

    auto C =
        make_identity_tensor(
            make_shape(Int<16>{}, Int<8>{})
        );

    printf("\nC mapping:\n");

    int lanes[] = {0, 1, 2, 3, 8, 16, 31};

    for (int lane : lanes) {
        auto thr_mma =
            tiled_mma.get_slice(lane);

        auto tC =
            thr_mma.partition_C(C);

        printf("Lane %d, shape = ", lane);
        print(shape(tC));
        printf("\n");

        for (int i = 0; i < size(tC); ++i) {
            auto coord = tC(i);

            auto m = get<0>(coord);
            auto n = get<1>(coord);

            printf(
                "  V%d -> C(%d, %d)\n",
                i,
                int(m),
                int(n)
            );
        }

        printf("\n");
    }
    return 0;
}
