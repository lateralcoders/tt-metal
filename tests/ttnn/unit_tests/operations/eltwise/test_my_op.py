import pytest
import torch
import ttnn


def test_my_op(device):
    input_tensor_a = ttnn.from_torch(torch.randn(1, 1, 128, 128)).to(device)
    input_tensor_b = ttnn.from_torch(torch.randn(1, 1, 128, 128)).to(device)
    scalar = 2

    output_tensor = ttnn.my_operation(input_tensor_a, input_tensor_b, scalar)
