import pytest
import torch
import ttnn


def test_my_op(device):
    input_tensor_a = torch.randn(1, 1, 128, 128)
    input_tensor_b = torch.randn(1, 1, 128, 128)

    tt_input_tensor_a = ttnn.from_torch(
        input_tensor_a,
        ttnn.bfloat16,
        pad_value=0,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_input_tensor_b = ttnn.from_torch(
        input_tensor_a,
        ttnn.bfloat16,
        pad_value=0,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scalar = 2

    tt_output_tensor = ttnn.my_operation(tt_input_tensor_a, tt_input_tensor_b, scalar)
