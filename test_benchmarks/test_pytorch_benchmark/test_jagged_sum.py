# import pytest
# from benchmark_utils import parse_torchbench_args
# from torchbenchmark.operators import load_opbench_by_name
# import torch


# @pytest.mark.parametrize("iter", range(1))
# def test_triton_jagged_sum_no_pad_simple_fused(iter):
#     Operator = load_opbench_by_name('jagged_sum')
#     opbench = Operator(tb_args=parse_torchbench_args())

#     nt, B, M, max_seqlen, sparsity = opbench.get_example_inputs()

#     ans = opbench.triton_jagged_sum_no_pad_simple_fused(nt, B, M, max_seqlen, sparsity)()

#     assert ans is not None, "ans is None"
#     assert ans.device.type == 'cuda'

# @pytest.mark.parametrize("iter", range(1))
# def test_triton_jagged_sum_no_pad_variable_length_loop(iter):
#     Operator = load_opbench_by_name('jagged_sum')
#     opbench = Operator(tb_args=parse_torchbench_args())

#     nt, B, M, max_seqlen, sparsity = opbench.get_example_inputs()

#     ans = opbench.triton_jagged_sum_no_pad_variable_length_loop(nt, B, M, max_seqlen, sparsity)()

#     assert ans is not None, "ans is None"
#     assert ans.device.type == 'cuda'