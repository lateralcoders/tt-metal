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

void bind_my_operation(py::module& module);

}  // namespace ttnn::operations::examples
