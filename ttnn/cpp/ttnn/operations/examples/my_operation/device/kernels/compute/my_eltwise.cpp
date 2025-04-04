#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "debug/dprint.h"

ALWI void sqrt_add_mul(uint32_t num_of_tiles) {
    DPRINT << "Doing op for " << num_of_tiles << " tiles" << ENDL();

    const uint32_t scalar = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = get_compile_time_arg_val(3);
    constexpr auto cb_in1 = get_compile_time_arg_val(4);

    constexpr auto cb_out0 = get_compile_time_arg_val(5);

    constexpr auto cb_inp0 = get_compile_time_arg_val(6);

    // copy B to dest
    DPRINT << "B to Dest" << ENDL();
    cb_wait_front(cb_in1, num_of_tiles);
    ckernel::tile_regs_acquire();
    ckernel::copy_tile_init(cb_in1);
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::copy_tile(cb_in1, i, i);
    }

    // multiply by scalar
    DPRINT << "Multiplying" << ENDL();
    ckernel::binop_with_scalar_tile_init();
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::mul_unary_tile(i, scalar);
    }
    ckernel::tile_regs_commit();
    cb_pop_front(cb_in1, num_of_tiles);

    // push to intermediate buffer C
    DPRINT << "Pushing to C" << ENDL();
    cb_reserve_back(cb_inp0, num_of_tiles);
    ckernel::tile_regs_wait();
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::pack_tile(i, cb_inp0, i);
    }
    ckernel::tile_regs_release();
    cb_push_back(cb_inp0, num_of_tiles);

    // add A and C and take sqrt
    DPRINT << "Adding A and C and taking sqrt" << ENDL();
    cb_wait_front(cb_in0, num_of_tiles);
    cb_wait_front(cb_inp0, num_of_tiles);
    ckernel::tile_regs_acquire();

    ckernel::add_tiles_init(cb_in0, cb_inp0);
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::add_tiles(cb_in0, cb_inp0, i, i, i);
    }

    ckernel::sqrt_tile_init();
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::sqrt_tile(i);
    }

    ckernel::tile_regs_commit();
    cb_pop_front(cb_in0, num_of_tiles);
    cb_pop_front(cb_inp0, num_of_tiles);

    // move to out
    DPRINT << "Moving to output" << ENDL();
    cb_reserve_back(cb_out0, num_of_tiles);
    ckernel::tile_regs_wait();
    for (uint32_t i = 0; i < num_of_tiles; i++) {
        ckernel::pack_tile(i, cb_out0, i);
    }
    ckernel::tile_regs_release();
    cb_push_back(cb_out0, num_of_tiles);
}

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);

    uint32_t full_blocks = per_core_tile_cnt / per_core_block_size;
    uint32_t remainder = per_core_tile_cnt % per_core_block_size;

    DPRINT << "Compute params:" << ENDL() << "per_core_tile_cnt: " << per_core_tile_cnt << ENDL()
           << "per_core_block_size: " << per_core_block_size << ENDL() << "full_blocks: " << full_blocks << ENDL()
           << "remainder: " << remainder << ENDL();

    for (uint32_t i = 0; i < full_blocks; i++) {
        sqrt_add_mul(per_core_block_size);
    }
    sqrt_add_mul(remainder);

    DPRINT << "Compute done" << ENDL();
}
}  // namespace NAMESPACE
