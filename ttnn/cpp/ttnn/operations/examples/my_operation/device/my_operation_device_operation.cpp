// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "my_operation_device_operation.hpp"
#include "tt-metalium/assert.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::examples {

MyDeviceOperation::program_factory_t MyDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MyDeviceProgramFactory{};
}

void MyDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    tt::tt_metal::TensorLayout layout_a = tensor_args.input_tensor_a.get_tensor_spec().tensor_layout();
    tt::tt_metal::TensorLayout layout_b = tensor_args.input_tensor_b.get_tensor_spec().tensor_layout();

    TT_ASSERT(layout_a == layout_b, "MyDeviceOperation requires both input tensors to have the same layout.");
    TT_ASSERT(
        layout_a.get_memory_config().is_dram() && layout_b.get_memory_config().is_dram(),
        "MyDeviceOperation only supports DRAM for input tensors.");
    TT_ASSERT(
        !layout_a.get_memory_config().is_sharded() && !layout_b.get_memory_config().is_sharded(),
        "MyDeviceOperation does not support sharded tensors.");
    TT_ASSERT(
        layout_a.get_layout() == tt::tt_metal::Layout::TILE && layout_b.get_layout() == tt::tt_metal::Layout::TILE,
        "MyDeviceOperation only supports TILE layout for input tensors.");
}

void MyDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate the operation attributes and tensor arguments here
}

MyDeviceOperation::spec_return_value_t MyDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    return TensorSpec(input_tensor_a.get_tensor_spec());
}

MyDeviceOperation::tensor_return_value_t MyDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
}

std::tuple<MyDeviceOperation::operation_attributes_t, MyDeviceOperation::tensor_args_t> MyDeviceOperation::invoke(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, const int input_scalar) {
    return {operation_attributes_t{input_scalar}, tensor_args_t{input_tensor_a, input_tensor_b}};
}

}  // namespace ttnn::operations::examples
