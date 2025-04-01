// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "my_operation_device_operation.hpp"
#include "tt-metalium/bfloat16.hpp"

namespace ttnn::opearations::examples {

void MyDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate the operation attributes and tensor arguments here
}

void MyDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate the operation attributes and tensor arguments here
}

MyDeviceOperation::program_spec_t MyDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_arg) {
    // Compute the output specifications based on the input tensor
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    return TensorSpec(
        input_tensor_a.get_logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor_a.get_dtype(), tt::tt_metal::PageConfig(input_tensor_a.get_layout()), MemoryConfig{}));
}

MyDeviceOperation::tensor_return_value_t MyDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Create output tensors based on the input tensors
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor_a.device());
}

std::tuple<MyDeviceOperation::operation_attributes_t, MyDeviceOperation::tensor_args_t> MyDeviceOperation::invoke(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, bfloat16 input_scalar) {
    // Invoke the operation with the given input tensors
    return {operation_attributes_t{input_scalar}, tensor_args_t{input_tensor_a, input_tensor_b}};
}

}  // namespace ttnn::opearations::examples
