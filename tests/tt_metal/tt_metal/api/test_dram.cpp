// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "dispatch_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/command_queue.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/logger.hpp>

#include <variant>
#include <optional>

using namespace tt;
using namespace tt::tt_metal;

namespace unit_tests_common::dram::test_dram {
struct DRAMConfig {
    // CoreRange, Kernel, dram_buffer_size
    CoreRange core_range;
    std::string kernel_file;
    std::uint32_t dram_buffer_size;
    std::uint32_t l1_buffer_addr;
    std::variant<tt_metal::DataMovementConfig, tt_metal::EthernetConfig> kernel_cfg;
    std::optional<uint32_t> dram_channel;
};

bool dram_single_core_db(tt::tt_metal::DispatchFixture* fixture, tt_metal::IDevice* device) {
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 256;
    uint32_t dram_buffer_size_bytes = single_tile_size * num_tiles;

    // L1 buffer is double buffered
    // We read and write total_l1_buffer_size_tiles / 2 tiles from and to DRAM
    uint32_t l1_buffer_addr = 400 * 1024;
    uint32_t total_l1_buffer_size_tiles = num_tiles / 2;
    TT_FATAL(total_l1_buffer_size_tiles % 2 == 0, "Error");
    uint32_t total_l1_buffer_size_bytes = total_l1_buffer_size_tiles * single_tile_size;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size_bytes,
        .page_size = dram_buffer_size_bytes,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();

    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();

    auto dram_copy_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_db.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
        dram_buffer_size_bytes, 100, std::chrono::system_clock::now().time_since_epoch().count());
    fixture->WriteBuffer(device, input_dram_buffer, input_vec);

    tt_metal::SetRuntimeArgs(
        program,
        dram_copy_kernel,
        core,
        {input_dram_buffer_addr,
        (std::uint32_t)0,
        output_dram_buffer_addr,
        (std::uint32_t)0,
        dram_buffer_size_bytes,
        num_tiles,
        l1_buffer_addr,
        total_l1_buffer_size_tiles,
        total_l1_buffer_size_bytes});

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, output_dram_buffer, result_vec);

    return input_vec == result_vec;
}

bool dram_single_core(
    tt::tt_metal::DispatchFixture* fixture,
    tt_metal::IDevice* device,
    const DRAMConfig& cfg,
    std::vector<uint32_t> src_vec) {
    // Create a program
    tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = cfg.dram_buffer_size,
        .page_size = cfg.dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    auto input_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();

    auto output_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();

    // zero out the output buffer
    std::vector<uint32_t> zero_vec(src_vec.size(), 0);
    tt_metal::detail::WriteToBuffer(output_dram_buffer, zero_vec);
    tt::Cluster::instance().dram_barrier(device->id());

    log_debug(tt::LogVerif, "Creating kernel");
    // Create the kernel
    KernelHandle dram_kernel;
    if (std::holds_alternative<tt_metal::DataMovementConfig>(cfg.kernel_cfg)) {
        dram_kernel = tt_metal::CreateKernel(
            program, cfg.kernel_file, cfg.core_range, std::get<tt_metal::DataMovementConfig>(cfg.kernel_cfg));
    } else {
        std::cout << "Creating eth kernel on core " << cfg.core_range.str() << std::endl;
        dram_kernel = tt_metal::CreateKernel(
            program, cfg.kernel_file, cfg.core_range, std::get<tt_metal::EthernetConfig>(cfg.kernel_cfg));
    }

    if (cfg.dram_channel.has_value()) {
        TT_FATAL(
            cfg.dram_channel.value() < device->num_dram_channels(),
            "Illegal dram channel {}",
            cfg.dram_channel.value());
        tt_metal::detail::WriteToDeviceDRAMChannel(device, cfg.dram_channel.value(), input_dram_buffer_addr, src_vec);
    } else {
        fixture->WriteBuffer(device, input_dram_buffer, src_vec);
    }

    uint32_t bank = cfg.dram_channel.has_value() ? cfg.dram_channel.value() : 0;

    tt_metal::SetRuntimeArgs(
        program,
        dram_kernel,
        cfg.core_range,
        {cfg.l1_buffer_addr, input_dram_buffer_addr, bank, output_dram_buffer_addr, bank, cfg.dram_buffer_size});

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec;
    // if (cfg.dram_channel.has_value()) {
    //     fixture->ReadBuffer(device, output_dram_buffer, result_vec);
    // }
    // else {
    //     tt_metal::detail::ReadFromDeviceDRAMChannel(
    //         device, cfg.dram_channel.value(), output_dram_buffer_addr, cfg.dram_buffer_size, result_vec);
    // }

    // if (result_vec != src_vec) {
    //     for (int i = 0; i < result_vec.size(); i++) {
    //         std::cout << "Expected " << src_vec.at(i) << " got " << result_vec.at(i) << std::endl;
    //     }
    // }
    return result_vec == src_vec;
}

bool dram_single_core_pre_allocated(
    tt::tt_metal::DispatchFixture* fixture,
    tt_metal::IDevice* device,
    const DRAMConfig& cfg,
    std::vector<uint32_t> src_vec) {
    // Create a program
    tt_metal::Program program = tt::tt_metal::CreateProgram();

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = cfg.dram_buffer_size,
        .page_size = cfg.dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto input_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t input_dram_buffer_addr = input_dram_buffer->address();
    auto input_dram_pre_allocated_buffer = tt_metal::CreateBuffer(dram_config, input_dram_buffer_addr);
    uint32_t input_dram_pre_allocated_buffer_addr = input_dram_pre_allocated_buffer->address();

    EXPECT_EQ(input_dram_buffer_addr, input_dram_pre_allocated_buffer_addr);

    auto output_dram_buffer = tt_metal::CreateBuffer(dram_config);
    uint32_t output_dram_buffer_addr = output_dram_buffer->address();
    auto output_dram_pre_allocated_buffer = tt_metal::CreateBuffer(dram_config, output_dram_buffer_addr);
    uint32_t output_dram_pre_allocated_buffer_addr = output_dram_pre_allocated_buffer->address();

    EXPECT_EQ(output_dram_buffer_addr, output_dram_pre_allocated_buffer_addr);

    // Create the kernel
    KernelHandle dram_kernel;
    if (std::holds_alternative<tt_metal::DataMovementConfig>(cfg.kernel_cfg)) {
        dram_kernel = tt_metal::CreateKernel(
            program, cfg.kernel_file, cfg.core_range, std::get<tt_metal::DataMovementConfig>(cfg.kernel_cfg));
    } else {
        dram_kernel = tt_metal::CreateKernel(
            program, cfg.kernel_file, cfg.core_range, std::get<tt_metal::EthernetConfig>(cfg.kernel_cfg));
    }
    fixture->WriteBuffer(device, input_dram_pre_allocated_buffer, src_vec);

    tt_metal::SetRuntimeArgs(
        program,
        dram_kernel,
        cfg.core_range,
        {cfg.l1_buffer_addr,
         input_dram_pre_allocated_buffer_addr,
         (std::uint32_t)0,
         output_dram_pre_allocated_buffer_addr,
         (std::uint32_t)0,
         cfg.dram_buffer_size});

    fixture->RunProgram(device, program);

    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, output_dram_pre_allocated_buffer, result_vec);

    return result_vec == src_vec;
}

}  // namespace unit_tests_common::dram::test_dram

namespace tt::tt_metal {

TEST_F(DispatchFixture, TensixDRAMLoopbackSingleCore) {
    uint32_t buffer_size = 2 * 1024 * 25;
    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = 400 * 1024,
        .kernel_cfg =
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
    };
    for (unsigned int id = 0; id < devices_.size(); id++) {
        ASSERT_TRUE(
            unit_tests_common::dram::test_dram::dram_single_core(this, devices_.at(id), dram_test_config, src_vec));
    }
}

TEST_F(DispatchFixture, TensixDRAMLoopbackSingleCorePreAllocated) {
    uint32_t buffer_size = 2 * 1024 * 25;
    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = 400 * 1024,
        .kernel_cfg =
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
    };
    for (unsigned int id = 0; id < devices_.size(); id++) {
        ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core_pre_allocated(
            this, devices_.at(id), dram_test_config, src_vec));
    }
}

TEST_F(DispatchFixture, TensixDRAMLoopbackSingleCoreDB) {
    if (!this->IsSlowDispatch()) {
        tt::log_info(tt::LogTest, "This test is only supported in slow dispatch mode");
        GTEST_SKIP();
    }
    for (unsigned int id = 0; id < devices_.size(); id++) {
        ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core_db(this, devices_.at(id)));
    }
}

TEST_F(DispatchFixture, ActiveEthDRAMLoopbackSingleCore) {
    uint32_t buffer_size = 2 * 1024 * 25;
    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = 0,
        .kernel_cfg = tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0},
    };
    for (unsigned int id = 0; id < devices_.size(); id++) {
        auto active_eth_core = *(devices_.at(id)->get_active_ethernet_cores(true).begin());
        for (auto active_eth_core : devices_.at(id)->get_active_ethernet_cores(true)) {
            dram_test_config.core_range = {active_eth_core, active_eth_core};
            dram_test_config.l1_buffer_addr =
                hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
            ASSERT_TRUE(
                unit_tests_common::dram::test_dram::dram_single_core(this, devices_.at(id), dram_test_config, src_vec));
        }
    }
}

TEST_F(DispatchFixture, IdleEthDRAMLoopbackSingleCore) {
    uint32_t buffer_size = 2 * 1024 * 25;
    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    unit_tests_common::dram::test_dram::DRAMConfig dram_test_config = {
        .core_range = {{0, 0}, {0, 0}},
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        .dram_buffer_size = buffer_size,
        .l1_buffer_addr = 0,
        .kernel_cfg = tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_0},
    };
    for (unsigned int id = 0; id < devices_.size(); id++) {
        for (unsigned int id = 0; id < devices_.size(); id++) {
            for (auto idle_eth_core : devices_.at(id)->get_inactive_ethernet_cores()) {
                dram_test_config.core_range = {idle_eth_core, idle_eth_core};
                dram_test_config.l1_buffer_addr =
                    hal.get_dev_size(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED);
                ASSERT_TRUE(unit_tests_common::dram::test_dram::dram_single_core(
                    this, devices_.at(id), dram_test_config, src_vec));
            }
        }
    }
}

}  // namespace tt::tt_metal
