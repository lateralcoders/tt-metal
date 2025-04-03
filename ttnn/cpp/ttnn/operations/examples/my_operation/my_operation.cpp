#include "ttnn/operations/examples/my_operation/my_operation.hpp"

namespace ttnn::operations::examples {
Tensor MyOperation::invoke(const Tensor& input_tensor_a, const Tensor& input_tensor_b, const int input_scalar) {
    return ttnn::prim::my_operation(input_tensor_a, input_tensor_b, input_scalar);
}
}  // namespace ttnn::operations::examples
