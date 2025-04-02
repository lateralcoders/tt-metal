// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// #include "debug/dprint.h"

// split REDUCE across cores
void kernel_main() {
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;

    uint32_t reduce_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(1));
    uint32_t reduce_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(2));

    constexpr uint32_t num_batch_group = get_compile_time_arg_val(3);

    constexpr uint32_t per_core_N = get_compile_time_arg_val(4);
    const uint32_t per_core_N_bytes = get_compile_time_arg_val(5);
    const uint32_t per_core_N_bytes_with_stride = get_compile_time_arg_val(6);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(7);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(8);

    volatile uint32_t block_h = get_compile_time_arg_val(9);
    constexpr uint32_t block_w = get_compile_time_arg_val(10);
    constexpr uint32_t block_hw = get_compile_time_arg_val(11);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_channels_tiles = get_arg_val<uint32_t>(2);
    const uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;  // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CBIndex::c_9;          // E[x] partial reduce
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;  // E[x] global reduce
    constexpr uint32_t cb_ex2 = tt::CBIndex::c_13;        // E[x]^2 partial reduce
    constexpr uint32_t cb_ex2_global = tt::CBIndex::c_14;  // E[x]^2 global reduce
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;         // input cb
    constexpr uint32_t cb_repack = tt::CBIndex::c_26;
    constexpr uint32_t cb_repack_out = tt::CBIndex::c_31;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    const uint32_t single_tile_size_bytes = get_tile_size(cb_ex_partial);  // tile size
    const DataFormat data_format = get_dataformat(cb_ex_partial);          // data format

    volatile tt_l1_ptr uint32_t* reduce_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* reduce_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduce_sender_semaphore_addr);

    const uint64_t reduce_receiver_semaphore_noc_addr =
        get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, reduce_receiver_semaphore_addr);

#if defined(READER_REPACK) and defined(TILIZE_IN)
    uint32_t in0_l1_read_addr = get_read_ptr(cb_in0);
    uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_reserve_back(cb_repack, per_core_N);
        uint32_t l1_write_addr_repack = get_write_ptr(cb_repack);
        for (uint32_t i = 0; i < TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes;
            l1_write_addr_repack += per_core_N_bytes_with_stride;
        }
        noc_async_read_barrier();
        cb_push_back(cb_repack, per_core_N);
    }
#endif

    uint32_t out_block_h = block_h / 1;  // TODO READ THIS IN FROM USER
    uint32_t out_block_hw = out_block_h * block_w;
    uint32_t num_out_blocks = block_h / out_block_h;

    for (uint32_t i = 0; i < num_batch_group; ++i) {
        for (uint32_t n = 0; n < 3; ++n) {
            uint32_t out_block_start_id_offset = 0;
            for (uint32_t out_block_index = 0; out_block_index < num_out_blocks; out_block_index++) {
#if !defined(READER_REPACK) or !defined(TILIZE_IN)
                const uint32_t src0_tile_bytes = get_tile_size(cb_in0);
                const DataFormat src0_data_format = get_dataformat(cb_in0);
                const InterleavedAddrGenFast<src0_is_dram> src_a = {
                    .bank_base_address = src_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};
                uint32_t l1_write_addr;
                if (n == 0) {
                    l1_write_addr = get_write_ptr(cb_in0);
                    cb_reserve_back(cb_in0, out_block_hw);
                } else if (n == 1 || n == 2) {
                    l1_write_addr = get_write_ptr(cb_x);
                    cb_reserve_back(cb_x, out_block_hw);
                }
                for (uint32_t mt = 0; mt < out_block_h; mt++) {
                    for (uint32_t nt = 0; nt < block_w; nt++) {
                        noc_async_read_tile(
                            start_id + out_block_start_id_offset + (mt * num_channels_tiles) + nt,
                            src_a,
                            l1_write_addr);
                        l1_write_addr += src0_tile_bytes;
                        noc_async_read_barrier();
                    }
                }
                out_block_start_id_offset += block_h * num_channels_tiles;
                if (n == 0) {
                    cb_push_back(cb_in0, out_block_hw);
                } else if (n == 1 || n == 2) {
                    cb_push_back(cb_x, out_block_hw);
                }
#endif
                if (n == 0 || n == 1) {
                    // wait for local data ready
                    noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
                    cb_wait_front(cb_ex_partial, 1);
                    noc_semaphore_inc(reduce_receiver_semaphore_noc_addr, 1);

                    noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
                    cb_pop_front(cb_ex_partial, 1);
                }
            }

            if (n == 0 || n == 1) {
                noc_semaphore_set(reduce_sender_semaphore_addr_ptr, INVALID);
                if (n == 0) {
                    cb_mcast_receive = cb_ex_global;
                } else if (n == 1) {
                    cb_mcast_receive = cb_ex2_global;
                }
                cb_reserve_back(cb_mcast_receive, 1);
                noc_semaphore_wait(reduce_sender_semaphore_addr_ptr, VALID);
                cb_push_back(cb_mcast_receive, 1);
            }
        }
    }

#if defined(READER_REPACK) and defined(UNTILIZE_OUT)
    uint32_t l1_write_addr_repack = get_write_ptr(cb_out0);
    for (uint32_t m = 0; m < per_core_M; ++m) {
        cb_wait_front(cb_repack_out, per_core_N);
        uint32_t in0_l1_read_addr = get_read_ptr(cb_repack_out);
        uint64_t noc_addr_in0 = get_noc_addr(in0_l1_read_addr);
        for (uint32_t i = 0; i < TILE_HEIGHT; ++i) {
            noc_async_read(noc_addr_in0, l1_write_addr_repack, per_core_N_bytes);
            noc_addr_in0 += per_core_N_bytes_with_stride;
            l1_write_addr_repack += per_core_N_bytes;
        }
        noc_async_read_barrier();
        cb_pop_front(cb_repack_out, per_core_N);
    }
#endif
}
