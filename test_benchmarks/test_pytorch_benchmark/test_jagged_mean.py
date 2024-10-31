import pytest
from benchmark_utils import parse_torchbench_args, check_out_of_bounds
from torchbenchmark.operators import load_opbench_by_name
import torch


@pytest.mark.parametrize("iter", range(1))
def test_triton_jagged_mean_simple_fused_sum_then_buffer(iter):
    Operator = load_opbench_by_name('jagged_mean')
    opbench = Operator(
        tb_args=parse_torchbench_args(),
        extra_args=["--sum-then-buffer", "1"]
    )

    nt, B, M, max_seqlen, sparsity = opbench.get_example_inputs()

    ans = opbench.triton_jagged_mean_simple_fused(nt, B, M, max_seqlen, sparsity)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'

    check_out_of_bounds()

@pytest.mark.parametrize("iter", range(1))
def test_triton_jagged_mean_simple_fused_buffer_then_sum(iter):
    Operator = load_opbench_by_name('jagged_mean')
    opbench = Operator(
        tb_args=parse_torchbench_args(),
        extra_args=["--sum-then-buffer", "0"]
    )

    nt, B, M, max_seqlen, sparsity = opbench.get_example_inputs()

    ans = opbench.triton_jagged_mean_simple_fused(nt, B, M, max_seqlen, sparsity)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'

    check_out_of_bounds()

@pytest.mark.parametrize("iter", range(1))
def test_triton_jagged_mean_variable_length_loop_sum_then_buffer(iter):
    Operator = load_opbench_by_name('jagged_mean')
    opbench = Operator(
        tb_args=parse_torchbench_args(),
        extra_args=["--sum-then-buffer", "1"])

    nt, B, M, max_seqlen, sparsity = opbench.get_example_inputs()

    ans = opbench.triton_jagged_mean_variable_length_loop(nt, B, M, max_seqlen, sparsity)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'

    check_out_of_bounds()

@pytest.mark.parametrize("iter", range(1))
def test_triton_jagged_mean_variable_length_loop_buffer_then_sum(iter):
    Operator = load_opbench_by_name('jagged_mean')
    opbench = Operator(
        tb_args=parse_torchbench_args(),
        extra_args=["--sum-then-buffer", "0"])

    nt, B, M, max_seqlen, sparsity = opbench.get_example_inputs()

    ans = opbench.triton_jagged_mean_variable_length_loop(nt, B, M, max_seqlen, sparsity)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'

    check_out_of_bounds()