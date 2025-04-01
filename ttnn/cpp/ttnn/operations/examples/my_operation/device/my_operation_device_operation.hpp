// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "tt-metalium/bfloat16.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::examples {
struct MyDeviceOperation {
    struct operation_attributes_t {
        bfloat16 input_scalar;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        const Tensor& input_tensor_b;
        ;
    }

    using spec_return_value_t = ttnn::TensorSpec;
    using tenaor_return_value_t = Tensor;

    struct MyDeviceProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            std::size_t num_cores_x;
            std::size_t num_cores_y;
        } using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    }  // struct MyDeviceProgramFactory;

    using program_factory_t = MyDeviceProgramFactory;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        return MyDeviceProgramFactory{};
    }

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(const& input_tensor_a, const& input_tensor_b);

};  // struct MyDeviceOperation
}  // namespace ttnn::operations::examples

namespace ttnn::prim {
constexpr auto my_device_operation =
    ttnn::register_operation<"ttnn::prim::MyDeviceOperation", ttnn::operations::examples::MyDeviceOperation>();
}  // namespace ttnn::prim
