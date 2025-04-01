// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::examples {
struct MyOperation {
    static Tensor invoke(const Tensor& input_tensor_a, const Tensor& input_tensor_b, bfloat16 input_scalar) {
        // TODO: call device operation and return
        return Tensor();
    }
};
}  // namespace ttnn::operations::examples

namespace ttnn {
constexpr auto my_operation = ttnn::register_operation<"ttnn::my_operation", operations::examples::MyOperation>();
}
