// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "dataflow_api.h"
#include "dataflow_api_addrgen.h"
#include "debug/dprint.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    DPRINT << "Writer args: " << ENDL() << "dst_addr: " << dst_addr << ENDL()
           << "input_start_tile_id: " << input_start_tile_id << ENDL() << "num_tiles: " << num_tiles << ENDL();

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_2;

    const uint32_t ublock_size_tiles = 1;

    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = dst_addr,
        .page_size = get_tile_size(cb_id_out0),
        .data_format = get_dataformat(cb_id_out0),
    };

    for (int i = 0; i < num_tiles; i += ublock_size_tiles) {
        uint32_t dst_tile_id = dst_start_tile_id;

        cb_wait_front(cb_id_out0, ublock_size_tiles);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

        noc_async_write_tile(dst_tile_id, s, l1_read_addr);
        noc_async_read_barrier();

        cb_pop_front(cb_id_out0, ublock_size_tiles);

        dst_tile_id += ublock_size_tiles;
    }
}
