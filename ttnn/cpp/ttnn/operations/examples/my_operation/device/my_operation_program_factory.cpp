#include <tt-metalium/work_split.hpp>
#include "my_operation_device_operation.hpp"

namespace ttnn::operations::examples {

MyDeviceOperation::MyDeviceProgramFactory::cached_program_t MyDeviceOperation::MyDeviceProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    auto& output_tensor = tensor_return_value;

    auto src_buffer_a = input_tensor_a.buffer();
    auto src_buffer_b = input_tensor_b.buffer();
    auto dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    // calculate tile sizes and number of tiles
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input_tensor_a.volume() / tt::constants::TILE_HW;

    // split work to cores
    tt::tt_metal::IDevice* device = input_tensor_a.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    // source and output buffers for the reader
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t output_cb_index = tt::CBIndex::c_2;

    uint32_t mul_cb_index = tt::CBIndex::c_3;

    uint32_t num_input_tiles = num_tiles * 2;
    uint32_t num_output_tiles = num_tiles * 2;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    tt::tt_metal::CircularBufferConfig cb_mul_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{mul_cb_index, cb_data_format}})
            .set_page_size(mul_cb_index, single_tile_size);
    auto cb_mul = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_mul_config);

    // create reader and writer kernels
    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index, src1_cb_index, 1};
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index, 1};

    tt::tt_metal::KernelHandle reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/my_operation/device/kernels/dataflow/my_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/my_operation/device/kernels/dataflow/my_writer.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // create compute kernel
    std::vector<uint32_t> compute_compile_time_args_1 = {
        num_tiles_per_core_group_1,
        16,
        0x40000000,  // 2.0f in uint32_t
        src0_cb_index,
        src1_cb_index,
        output_cb_index,
        mul_cb_index};
    std::vector<uint32_t> compute_compile_time_args_2 = {
        num_tiles_per_core_group_2,
        16,
        0x40000000,  // 2.0f in uint32_t
        src0_cb_index,
        src1_cb_index,
        output_cb_index,
        mul_cb_index};

    auto compute_kernel_1 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/my_operation/device/kernels/compute/my_eltwise.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .compile_args = compute_compile_time_args_1});

    auto compute_kernel_2 =
        (core_group_2.ranges().empty())
            ? compute_kernel_1
            : tt::tt_metal::CreateKernel(
                  program,
                  "ttnn/cpp/ttnn/operations/examples/my_operation/device/kernels/compute/my_eltwise.cpp",
                  core_group_2,
                  tt::tt_metal::ComputeConfig{
                      .math_fidelity = MathFidelity::HiFi4,
                      .math_approx_mode = false,
                      .compile_args = compute_compile_time_args_2});

    // calculate tile being written for each core
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {src_buffer_a->address(),
             num_tiles_written,
             src_buffer_b->address(),
             num_tiles_written,
             num_tiles_per_core});
        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernel, core, {dst_buffer->address(), num_tiles_written, num_tiles_per_core});

        num_tiles_written += num_tiles_per_core;
    }

    // return program
    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel,
         .writer_kernel_id = writer_kernel,
         .compute_kernel_id_1 = compute_kernel_1,
         .compute_kernel_id_2 = compute_kernel_2,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void MyDeviceOperation::MyDeviceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& compute_kernel_id_1 = cached_program.shared_variables.compute_kernel_id_1;
    auto& compute_kernel_id_2 = cached_program.shared_variables.compute_kernel_id_2;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    auto src_buffer_a = input_tensor_a.buffer();
    auto src_buffer_b = input_tensor_b.buffer();
    auto dst_buffer = output_tensor.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer_a->address();
            runtime_args[2] = src_buffer_b->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::examples
