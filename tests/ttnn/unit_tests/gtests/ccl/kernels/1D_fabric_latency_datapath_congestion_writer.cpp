// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "dataflow_api.h"

#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_transmission.hpp"

#include <cstdint>
#include <cstddef>
#include "debug/dprint.h"

constexpr bool use_mcast_mode = get_compile_time_arg_val(0) != 0;

void kernel_main() {
    using namespace tt::fabric;
    size_t arg_idx = 0;

    DPRINT << "Starting latency test writer kernel\n";

    const size_t dest_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const size_t dest_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const size_t dest_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t message_num_hops = get_arg_val<uint32_t>(arg_idx++);
    // We don't really care about the contents of the payloads for this microbenchmark,
    // so we reuse the output L1 address
    auto teardown_signal_addr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    const size_t packet_header_cb = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_size_in_headers = get_arg_val<uint32_t>(arg_idx++);
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    const size_t source_l1_buffer_address = dest_bank_addr;

    // ASSERT(fabric_connection.is_logically_connected());
    // if (!fabric_connection.is_logically_connected()) {
    //     return;
    // }

    // fabric_connection.open();
    // cb_reserve_back(packet_header_cb, packet_header_size_in_headers);
    // const auto packet_header_buffer_address = get_write_ptr(packet_header_cb);

    // auto* packet_header = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    // if constexpr (use_mcast_mode) {
    //     packet_header->to_chip_multicast(MulticastRoutingCommandHeader{1, static_cast<uint8_t>(message_num_hops)});
    // } else {
    //     packet_header->to_chip_unicast(static_cast<uint8_t>(message_num_hops));
    // }

    // auto noc0_dest_addr = safe_get_noc_addr(
    //     static_cast<uint8_t>(dest_noc_x), static_cast<uint8_t>(dest_noc_y), dest_bank_addr, 0);
    // auto dest_addr =
    //     safe_get_noc_addr(static_cast<uint8_t>(dest_noc_x), static_cast<uint8_t>(dest_noc_y), dest_bank_addr);
    // packet_header->to_noc_unicast_write(
    //     NocUnicastCommandHeader{noc0_dest_addr}, packet_payload_size_bytes);

    // while (*teardown_signal_addr == 0) {
    //     DPRINT << "Message \n";
    //     fabric_connection.get_forward_connection().wait_for_empty_write_slot();
    //     fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
    //         source_l1_buffer_address, packet_payload_size_bytes);
    //     fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
    //         (uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
    // }

    // fabric_connection.close();
    // noc_async_write_barrier();
}
