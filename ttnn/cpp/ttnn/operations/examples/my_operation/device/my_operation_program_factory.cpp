// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "my_operation_device_operation.hpp"

namespace ttnn::operations : examples {

    MyDeviceOperation::MyDeviceProgramFactory::create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {}

    MyDeviceOperation::MyDeviceProgramFactory::override_runtime_arguments(
        cached_program_t & cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {}

}  // namespace ttnn::operations
