// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "quantization.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"

#include <cassert>

namespace {

ttnn::DataType get_output_dtype(
    const std::optional<const ttnn::DataType>& output_dtype,
    const std::optional<ttnn::Tensor>& output_tensor,
    const ttnn::DataType default_dtype) {
    if (output_dtype.has_value() && output_tensor.has_value()) {
        TT_FATAL(output_dtype.value() == output_tensor->dtype(), "Mismatching output_dtype and output tensor dtype");
        return output_dtype.value();
    } else if (output_dtype.has_value()) {
        return output_dtype.value();
    } else if (output_tensor.has_value()) {
        return output_tensor->dtype();
    } else {
        return default_dtype;
    }
}

void check_per_tensor_scale_tensor(const ttnn::Tensor& scale) {
    const auto dtype = scale.get_dtype();
    TT_FATAL(tt::tt_metal::is_floating_point(dtype), "Quantization only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(dtype), "Unsupported quantization scale data type");
    TT_FATAL(scale.get_logical_volume() == 1u, "Per-tensor quantization only takes scalar-tensor scales");
}

void check_per_tensor_zero_point_tensor(const ttnn::Tensor& zero_point) {
    const auto dtype = zero_point.get_dtype();
    TT_FATAL(dtype == ttnn::DataType::INT32, "Quantization only takes int32 zero-points for now");
    TT_FATAL(zero_point.get_logical_volume() == 1u, "Per-tensor quantization only takes scalar-tensor zero-points");
}

void check_per_channel_tensor_args(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor* scale_p,
    const ttnn::Tensor* zero_point_p,
    const int32_t axis,
    const int32_t rank) {
    TT_FATAL(
        scale_p != nullptr && scale_p->get_logical_shape().rank() == 1,
        "Per-channel quantization expects 1D scale tensors");
    TT_FATAL(
        zero_point_p != nullptr && zero_point_p->get_logical_shape().rank() == 1,
        "Per-channel quantization expects 1D zero-point tensors");
    TT_FATAL(
        scale_p->get_logical_shape() == zero_point_p->get_logical_shape(),
        "Per-channel quantization expects scale & zero-point tensors of matching shapes");
    TT_FATAL(axis >= -rank && axis < rank, "Axis {} is outside the range [{}, {}]", axis, -rank, rank - 1);
    TT_FATAL(
        input_tensor.get_logical_shape()[axis] == scale_p->get_logical_volume(),
        "Size of the scale tensor doesn't match the size of the input tensor along the given axis");
    TT_FATAL(
        input_tensor.get_logical_shape()[axis] == zero_point_p->get_logical_volume(),
        "Size of the zero-point tensor doesn't match the size of the input tensor along the given axis");

    const auto scale_dtype = scale_p->get_dtype();
    TT_FATAL(tt::tt_metal::is_floating_point(scale_dtype), "Quantization only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(scale_dtype), "Unsupported quantization scale data type");

    const auto zero_point_dtype = zero_point_p->get_dtype();
    TT_FATAL(zero_point_dtype == ttnn::DataType::INT32, "Quantization only takes int32 zero-points for now");
}

ttnn::Tensor unsqueeze_per_channel_tensor_args(
    const ttnn::Tensor& input, const int32_t rank, const int32_t axis, const ttnn::DataType out_dtype) {
    // This function is internal use only, use asserts instead of TT_FATAL to convey intented usage
    assert(input.get_logical_shape().rank() == 1);
    assert(axis >= -rank && axis < rank);
    const int32_t axis_normalized = (axis + rank) % rank;
    // TODO: unsqueeze only supports u32/f32/bf16 so we must cast the unsupported inputs to some fp type
    ttnn::Tensor result = ttnn::typecast(input, out_dtype);
    for (int32_t i = 0; i < axis_normalized; i++) {
        result = ttnn::unsqueeze(result, 0);
    }
    for (int32_t i = axis_normalized + 1; i < rank; i++) {
        result = ttnn::unsqueeze(result, -1);
    }
    assert(result.get_logical_shape().rank() == rank);
    assert(result.get_logical_shape()[axis] == input.get_logical_volume());
    assert(result.get_logical_volume() == input.get_logical_volume());
    return result;
}

}  // anonymous namespace

namespace ttnn::operations::quantization {

Tensor QuantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& scale,
    const std::variant<Tensor, int32_t>& zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const Tensor input_a = tt::tt_metal::is_block_float(input_tensor.get_dtype())
                               ? ttnn::typecast(input_tensor, DataType::BFLOAT16)
                               : input_tensor;

    const DataType a_dtype = input_a.get_dtype();
    constexpr DataType c_dtype = DataType::INT32;

    TT_FATAL(tt::tt_metal::is_floating_point(a_dtype), "Quantize only takes floating-point number inputs");
    TT_FATAL(output_dtype.value_or(c_dtype) == c_dtype, "Quantize only supports int32 outputs for now");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(optional_output_tensor->dtype() == c_dtype, "Quantize only supports int32 outputs for now");
    }

    tt::stl::Span<const unary::UnaryWithParam> noop_activation{};

    const bool is_per_channel = axis.has_value();
    if (is_per_channel) {
        const Tensor* scale_p = std::get_if<Tensor>(&scale);
        const Tensor* zero_point_p = std::get_if<Tensor>(&zero_point);

        const int32_t axis_v = axis.value();
        const int32_t rank = static_cast<int32_t>(input_a.get_logical_shape().rank());

        check_per_channel_tensor_args(input_a, scale_p, zero_point_p, axis_v, rank);

        // TODO: fix the cast to a_dtype once binary ops get better support for mixed dtypes
        const Tensor scale_full = unsqueeze_per_channel_tensor_args(*scale_p, rank, axis_v, a_dtype);
        const Tensor zero_point_full = unsqueeze_per_channel_tensor_args(*zero_point_p, rank, axis_v, a_dtype);
        const Tensor input_scaled = ttnn::prim::binary_ng(
            queue_id,
            input_a,
            scale_full,
            binary_ng::BinaryOpType::DIV,
            a_dtype,
            std::nullopt,
            std::nullopt,
            noop_activation,
            noop_activation,
            noop_activation);
        return ttnn::typecast(
            ttnn::prim::binary_ng(
                queue_id,
                input_scaled,
                zero_point_full,
                binary_ng::BinaryOpType::ADD,
                std::nullopt,
                memory_config,
                optional_output_tensor,
                noop_activation,
                noop_activation,
                noop_activation),
            c_dtype);
    }

    return std::visit(
        [&](auto&& scale_v, auto&& zero_point_v) -> Tensor {
            constexpr bool scale_is_scalar = std::is_same_v<float, std::decay_t<decltype(scale_v)>>;
            constexpr bool zero_point_is_scalar = std::is_same_v<int32_t, std::decay_t<decltype(zero_point_v)>>;
            if constexpr (!scale_is_scalar) {
                check_per_tensor_scale_tensor(scale_v);
            }
            if constexpr (!zero_point_is_scalar) {
                check_per_tensor_zero_point_tensor(zero_point_v);
            }

            if constexpr (scale_is_scalar && zero_point_is_scalar) {
                const std::array post_activation{
                    unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(zero_point_v)}};

                // LLK quant kernel expects the reciprocal of the actual scale to avoid doing div on the device
                return ttnn::prim::binary_ng(
                    queue_id,
                    input_a,
                    1.0f / scale_v,
                    binary_ng::BinaryOpType::QUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    noop_activation,
                    noop_activation,
                    post_activation);
            } else if constexpr (zero_point_is_scalar) {
                const std::array post_activation{
                    unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(zero_point_v)}};

                // TODO: do reciprocal using an activation containing an UnaryWithParam that does rdiv(scale, 1)
                return ttnn::prim::binary_ng(
                    queue_id,
                    input_a,
                    ttnn::reciprocal(scale_v),
                    binary_ng::BinaryOpType::QUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    noop_activation,
                    noop_activation,
                    post_activation);
            } else if constexpr (scale_is_scalar) {
                // TODO: initialize the SFPU from the scalar-tensor to avoid this composite op fallback
                const Tensor input_scaled = ttnn::divide(input_a, scale_v);
                return ttnn::typecast(
                    ttnn::prim::binary_ng(
                        queue_id,
                        input_scaled,
                        zero_point_v.get_dtype() == a_dtype ? zero_point_v : ttnn::typecast(zero_point_v, a_dtype),
                        binary_ng::BinaryOpType::ADD,
                        a_dtype,
                        std::nullopt,
                        std::nullopt,
                        noop_activation,
                        noop_activation,
                        noop_activation),
                    c_dtype);
            } else {
                // TODO: initialize the SFPU from the scalar-tensor to avoid this composite op fallback
                const Tensor input_scaled = ttnn::prim::binary_ng(
                    queue_id,
                    input_a,
                    scale_v.get_dtype() == a_dtype ? scale_v : ttnn::typecast(scale_v, a_dtype),
                    binary_ng::BinaryOpType::DIV,
                    a_dtype,
                    std::nullopt,
                    std::nullopt,
                    noop_activation,
                    noop_activation,
                    noop_activation);
                return ttnn::typecast(
                    ttnn::prim::binary_ng(
                        queue_id,
                        input_scaled,
                        zero_point_v.get_dtype() == a_dtype ? zero_point_v : ttnn::typecast(zero_point_v, a_dtype),
                        binary_ng::BinaryOpType::ADD,
                        a_dtype,
                        std::nullopt,
                        std::nullopt,
                        noop_activation,
                        noop_activation,
                        noop_activation),
                    c_dtype);
            }
        },
        scale,
        zero_point);
}

Tensor RequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& in_scale,
    const std::variant<Tensor, int32_t>& in_zero_point,
    const std::variant<Tensor, float>& out_scale,
    const std::variant<Tensor, int32_t>& out_zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const DataType a_dtype = input_tensor.get_dtype();
    constexpr DataType c_dtype = DataType::INT32;

    TT_FATAL(a_dtype == DataType::INT32, "Requantize only supports int32 inputs for now");
    TT_FATAL(output_dtype.value_or(c_dtype) == c_dtype, "Requantize only supports int32 outputs for now");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(optional_output_tensor->dtype() == c_dtype, "Requantize only supports int32 outputs for now");
    }

    tt::stl::Span<const unary::UnaryWithParam> noop_activation{};

    const bool is_per_channel = axis.has_value();
    if (is_per_channel) {
        const Tensor* in_scale_p = std::get_if<Tensor>(&in_scale);
        const Tensor* in_zero_point_p = std::get_if<Tensor>(&in_zero_point);
        const Tensor* out_scale_p = std::get_if<Tensor>(&out_scale);
        const Tensor* out_zero_point_p = std::get_if<Tensor>(&out_zero_point);

        const int32_t axis_v = axis.value();
        const int32_t rank = static_cast<int32_t>(input_tensor.get_logical_shape().rank());

        check_per_channel_tensor_args(input_tensor, in_scale_p, in_zero_point_p, axis_v, rank);
        check_per_channel_tensor_args(input_tensor, out_scale_p, out_zero_point_p, axis_v, rank);

        // TODO: fix the cast to f32 once binary ops get better support for mixed dtypes
        const Tensor in_scale_full = unsqueeze_per_channel_tensor_args(*in_scale_p, rank, axis_v, DataType::FLOAT32);
        const Tensor in_zero_point_full =
            unsqueeze_per_channel_tensor_args(*in_zero_point_p, rank, axis_v, DataType::FLOAT32);
        const Tensor out_scale_full = unsqueeze_per_channel_tensor_args(*out_scale_p, rank, axis_v, DataType::FLOAT32);
        const Tensor out_zero_point_full =
            unsqueeze_per_channel_tensor_args(*out_zero_point_p, rank, axis_v, DataType::FLOAT32);

        const Tensor scale_recip_full = ttnn::prim::binary_ng(
            queue_id,
            in_scale_full,
            out_scale_full,
            binary_ng::BinaryOpType::DIV,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            noop_activation,
            noop_activation,
            noop_activation);
        const Tensor in_zero_point_scaled_full = ttnn::prim::binary_ng(
            queue_id,
            in_zero_point_full,
            scale_recip_full,
            binary_ng::BinaryOpType::MUL,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            noop_activation,
            noop_activation,
            noop_activation);
        const Tensor zero_point_full = ttnn::prim::binary_ng(
            queue_id,
            out_zero_point_full,
            in_zero_point_scaled_full,
            binary_ng::BinaryOpType::SUB,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            noop_activation,
            noop_activation,
            noop_activation);

        const Tensor input_scaled = ttnn::prim::binary_ng(
            queue_id,
            ttnn::typecast(input_tensor, DataType::FLOAT32),
            scale_recip_full,
            binary_ng::BinaryOpType::MUL,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            noop_activation,
            noop_activation,
            noop_activation);
        return ttnn::typecast(
            ttnn::prim::binary_ng(
                queue_id,
                input_scaled,
                zero_point_full,
                binary_ng::BinaryOpType::ADD,
                std::nullopt,
                memory_config,
                optional_output_tensor,
                noop_activation,
                noop_activation,
                noop_activation),
            c_dtype);
    }

    return std::visit(
        [&](auto&& in_scale_v, auto&& in_zero_point_v, auto&& out_scale_v, auto&& out_zero_point_v) -> Tensor {
            constexpr bool in_scale_is_scalar = std::is_same_v<float, std::decay_t<decltype(in_scale_v)>>;
            constexpr bool in_zero_point_is_scalar = std::is_same_v<int32_t, std::decay_t<decltype(in_zero_point_v)>>;
            constexpr bool out_scale_is_scalar = std::is_same_v<float, std::decay_t<decltype(out_scale_v)>>;
            constexpr bool out_zero_point_is_scalar = std::is_same_v<int32_t, std::decay_t<decltype(out_zero_point_v)>>;

            if constexpr (!in_scale_is_scalar) {
                check_per_tensor_scale_tensor(in_scale_v);
            }
            if constexpr (!in_zero_point_is_scalar) {
                check_per_tensor_zero_point_tensor(in_zero_point_v);
            }
            if constexpr (!out_scale_is_scalar) {
                check_per_tensor_scale_tensor(out_scale_v);
            }
            if constexpr (!out_zero_point_is_scalar) {
                check_per_tensor_zero_point_tensor(out_zero_point_v);
            }

            // Enable fast path for all scalar scales & zero-points, fallback to composite ops otherwise
            if constexpr (
                in_scale_is_scalar && in_zero_point_is_scalar && out_scale_is_scalar && out_zero_point_is_scalar) {
                // Expansion of q' = [(q - z_in) * s_in] / s_out + z_out
                const float scale_recip = in_scale_v / out_scale_v;
                // z is passed to and consumed by the LLK as f32 anyway, might as well preserve some accuracy here
                const float zero_point = out_zero_point_v - in_zero_point_v * scale_recip;

                const std::array post_activation{unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, zero_point}};
                return ttnn::prim::binary_ng(
                    queue_id,
                    input_tensor,
                    scale_recip,
                    binary_ng::BinaryOpType::REQUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    noop_activation,
                    noop_activation,
                    post_activation);
            } else {
                // The composite op fallback, generic but uses more ops and has worse accuracy
                const Tensor dequantized = DequantOp::invoke(
                    queue_id,
                    input_tensor,
                    in_scale_v,
                    in_zero_point_v,
                    axis,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt);
                return QuantOp::invoke(
                    queue_id,
                    dequantized,
                    out_scale_v,
                    out_zero_point_v,
                    axis,
                    c_dtype,
                    memory_config,
                    optional_output_tensor);
            }
        },
        in_scale,
        in_zero_point,
        out_scale,
        out_zero_point);
}

Tensor DequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& scale,
    const std::variant<Tensor, int32_t>& zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const DataType a_dtype = input_tensor.get_dtype();
    const DataType c_dtype = get_output_dtype(output_dtype, optional_output_tensor, DataType::BFLOAT16);

    TT_FATAL(a_dtype == DataType::INT32, "Dequantize only supports int32 inputs for now");
    TT_FATAL(
        c_dtype == DataType::FLOAT32 || c_dtype == DataType::BFLOAT16,
        "Dequantize only supports bf16/f32 outputs for now");

    tt::stl::Span<const unary::UnaryWithParam> noop_activation{};

    const bool is_per_channel = axis.has_value();
    if (is_per_channel) {
        const Tensor* scale_p = std::get_if<Tensor>(&scale);
        const Tensor* zero_point_p = std::get_if<Tensor>(&zero_point);

        const int32_t axis_v = axis.value();
        const int32_t rank = static_cast<int32_t>(input_tensor.get_logical_shape().rank());

        check_per_channel_tensor_args(input_tensor, scale_p, zero_point_p, axis_v, rank);

        // TODO: fix the cast to f32 once binary ops get better support for mixed dtypes
        const Tensor scale_full = unsqueeze_per_channel_tensor_args(*scale_p, rank, axis_v, DataType::FLOAT32);
        const Tensor zero_point_full =
            unsqueeze_per_channel_tensor_args(*zero_point_p, rank, axis_v, DataType::FLOAT32);
        const Tensor input_shifted = ttnn::prim::binary_ng(
            queue_id,
            ttnn::typecast(input_tensor, DataType::FLOAT32),
            zero_point_full,
            binary_ng::BinaryOpType::SUB,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            noop_activation,
            noop_activation,
            noop_activation);
        return ttnn::typecast(
            ttnn::prim::binary_ng(
                queue_id,
                input_shifted,
                scale_full,
                binary_ng::BinaryOpType::MUL,
                std::nullopt,
                memory_config,
                optional_output_tensor,
                noop_activation,
                noop_activation,
                noop_activation),
            c_dtype);
    }

    return std::visit(
        [&](auto&& scale_v, auto&& zero_point_v) -> Tensor {
            constexpr bool scale_is_scalar = std::is_same_v<float, std::decay_t<decltype(scale_v)>>;
            constexpr bool zero_point_is_scalar = std::is_same_v<int32_t, std::decay_t<decltype(zero_point_v)>>;
            if constexpr (!scale_is_scalar) {
                check_per_tensor_scale_tensor(scale_v);
            }
            if constexpr (!zero_point_is_scalar) {
                check_per_tensor_zero_point_tensor(zero_point_v);
            }

            if constexpr (zero_point_is_scalar) {
                // LLK dequant kernel does addition, so we need to negate zero_point
                const std::array post_activation{
                    unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(-zero_point_v)}};
                return ttnn::prim::binary_ng(
                    queue_id,
                    input_tensor,
                    scale_v,
                    binary_ng::BinaryOpType::DEQUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    noop_activation,
                    noop_activation,
                    post_activation);
            } else if constexpr (scale_is_scalar) {
                // TODO: initialize the SFPU from the scalar-tensor to avoid this composite op fallback
                const Tensor input_shifted = ttnn::typecast(
                    ttnn::prim::binary_ng(
                        queue_id,
                        input_tensor,
                        zero_point_v,
                        binary_ng::BinaryOpType::SUB,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        noop_activation,
                        noop_activation,
                        noop_activation),
                    c_dtype);
                return ttnn::multiply(input_shifted, scale_v);
            } else {
                // TODO: initialize the SFPU from the scalar-tensor to avoid this composite op fallback
                const Tensor input_shifted = ttnn::typecast(
                    ttnn::prim::binary_ng(
                        queue_id,
                        input_tensor,
                        zero_point_v,
                        binary_ng::BinaryOpType::SUB,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        noop_activation,
                        noop_activation,
                        noop_activation),
                    c_dtype);
                return ttnn::prim::binary_ng(
                    queue_id,
                    input_shifted,
                    scale_v.get_dtype() == c_dtype ? scale_v : ttnn::typecast(scale_v, c_dtype),
                    binary_ng::BinaryOpType::MUL,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    noop_activation,
                    noop_activation,
                    noop_activation);
            }
        },
        scale,
        zero_point);
}

}  // namespace ttnn::operations::quantization
