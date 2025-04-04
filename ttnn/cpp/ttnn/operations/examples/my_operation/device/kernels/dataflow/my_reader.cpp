// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "dataflow_api.h"
#include "dataflow_api_addrgen.h"
#include "debug/dprint.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t src1_start_tile_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    DPRINT << "Reader args: " << ENDL() << "src0_addr: " << src0_addr << ENDL()
           << "src0_start_tile_id: " << src0_start_tile_id << ENDL() << "src1_addr: " << src1_addr << ENDL()
           << "src1_start_tile_id: " << src1_start_tile_id << ENDL() << "num_tiles: " << num_tiles << ENDL();

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);

    const uint32_t ublock_size_tiles = get_compile_time_arg_val(2);

    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = src0_addr,
        .page_size = get_tile_size(cb_id_in0),
        .data_format = get_dataformat(cb_id_in0),
    };

    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = src1_addr,
        .page_size = get_tile_size(cb_id_in1),
        .data_format = get_dataformat(cb_id_in1),
    };

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint32_t src0_tile_id = src0_start_tile_id;
        uint32_t src1_tile_id = src1_start_tile_id;

        cb_reserve_back(cb_id_in0, ublock_size_tiles);
        cb_reserve_back(cb_id_in1, ublock_size_tiles);

        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        noc_async_read_tile(src0_tile_id, s0, l1_write_addr_in0);
        noc_async_read_tile(src1_tile_id, s1, l1_write_addr_in1);
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, ublock_size_tiles);
        cb_push_back(cb_id_in1, ublock_size_tiles);

        src0_tile_id += ublock_size_tiles;
        src1_tile_id += ublock_size_tiles;
    }

    DPRINT << "Reader done" << ENDL();
}
