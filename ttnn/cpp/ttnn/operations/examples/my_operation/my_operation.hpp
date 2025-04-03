// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/examples/my_operation/device/my_operation_device_operation.hpp"

namespace ttnn::operations::examples {
struct MyOperation {
    static Tensor invoke(const Tensor& input_tensor_a, const Tensor& input_tensor_b, const int input_scalar);
};
}  // namespace ttnn::operations::examples

namespace ttnn {
constexpr auto my_operation =
    ttnn::register_operation_with_auto_launch_op<"ttnn::my_operation", ttnn::operations::examples::MyOperation>();
}
