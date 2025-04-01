// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "ttnn/operations/examples/my_operation/my_operation.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::examples {

void bind_my_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::my_operation,
        R"doc(my_operation(a: ttnn.Tensor, b: ttnn>Tensor, scalar: bfloat16) -> ttnn.Tensor)doc",

        ttnn::pybind_overload_t{
            [](const decltype(ttnn::my_operation)& self,
               const ttnn::Tensor& input_tensor_a,
               const ttnn::Tensor input_tensor_b,
               bfloat16 input_scalar) -> ttnn::Tensor { return self(input_tensor_a, input_tensor_b, input_scalar); },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b"),
            py::arg("input_scalar")});
}

}  // namespace ttnn::operations::examples
