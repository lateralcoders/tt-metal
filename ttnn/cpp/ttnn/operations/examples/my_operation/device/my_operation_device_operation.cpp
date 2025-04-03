// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "my_operation_device_operation.hpp"

namespace ttnn::operations::examples {

MyDeviceOperation::program_factory_t MyDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return MyDeviceProgramFactory{};
}

void MyDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate the operation attributes and tensor arguments here
}

void MyDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Validate the operation attributes and tensor arguments here
}

MyDeviceOperation::spec_return_value_t MyDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    return TensorSpec(
        input_tensor_a.get_logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor_a.get_dtype(), tt::tt_metal::PageConfig(input_tensor_a.get_layout()), MemoryConfig{}));
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
