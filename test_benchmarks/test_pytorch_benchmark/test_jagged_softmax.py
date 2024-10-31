import pytest
import torch
from benchmark_utils import parse_torchbench_args, check_out_of_bounds
from torchbenchmark.operators import load_opbench_by_name

@pytest.mark.parametrize("iter", range(1))
def test_triton_jagged_softmax_simple_fused(iter):
    Operator = load_opbench_by_name('jagged_softmax')
    opbench = Operator(tb_args=parse_torchbench_args())

    nt, B, M, max_seqlen, sparsity = opbench.get_example_inputs()

    ans = opbench.triton_jagged_softmax_simple_fused(nt, B, M, max_seqlen, sparsity)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'

    check_out_of_bounds()

@pytest.mark.parametrize("iter", range(1))
def test_triton_jagged_softmax_variable_length_loop(iter):
    Operator = load_opbench_by_name('jagged_softmax')
    opbench = Operator(tb_args=parse_torchbench_args())

    nt, B, M, max_seqlen, sparsity = opbench.get_example_inputs()

    ans = opbench.triton_jagged_softmax_variable_length_loop(nt, B, M, max_seqlen, sparsity)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'

    check_out_of_bounds()