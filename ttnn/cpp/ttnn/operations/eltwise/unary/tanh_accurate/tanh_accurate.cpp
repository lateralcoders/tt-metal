// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tanh_accurate.hpp"

#include "ttnn/decorators.hpp"
#include "device/tanh_accurate_device_operation.hpp"
#include "ttnn/common/queue_id.hpp"
// #include "ttnn/run_operation.hpp"
// #include "ttnn/operations/pool/downsample/device/downsample_op.hpp"
// #include "ttnn/operations/core/core.hpp"

namespace ttnn {

namespace operations {

namespace unary {

Tensor Tanh_accurate::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    auto input_dtype = input_tensor.get_dtype();
    DataType output_dtype = input_dtype;
    auto arch = input_tensor.device()->arch();
    bool preserve_fp32_precision = (arch != tt::ARCH::GRAYSKULL) and (input_dtype == DataType::FLOAT32);

    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            input_dtype == DataType::UINT32 or input_dtype == DataType::INT32;

    bool bfp8_pack_precise = input_dtype == DataType::BFLOAT8_B;

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());

    return prim::tanh_accurate(
        queue_id,
        input_tensor,
        // op_chain,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor);
}

}  // namespace unary
}  // namespace operations
}  // namespace ttnn
