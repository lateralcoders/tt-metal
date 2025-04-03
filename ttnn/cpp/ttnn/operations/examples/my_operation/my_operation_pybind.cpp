// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/examples/my_operation/my_operation_pybind.hpp"

namespace ttnn::operations::examples {

void bind_my_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::my_operation,
        R"doc(my_operation(a: ttnn.Tensor, b: ttnn.Tensor, scalar: int) -> ttnn.Tensor)doc",
        ttnn::pybind_arguments_t{py::arg("input_tensor_a"), py::arg("input_tensor_b"), py::arg("input_scalar")});
}

}  // namespace ttnn::operations::examples
