import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name
import torch

@pytest.mark.parametrize("iter", range(31))
def test_triton_tutorial_matmul(iter):
    Operator = load_opbench_by_name('gemm')
    opbench = Operator(tb_args=parse_torchbench_args())
    
    a, b, bias = opbench.get_example_inputs()

    ans = opbench.triton_tutorial_matmul(a, b, bias)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'


# enable this test after fixing the bug.
# triton.runtime.errors.OutOfResources: out of resource:
# shared memory, Required: 115200, Hardware limit: 101376. Reducing block sizes or `num_stages` may help.

# @pytest.mark.parametrize("iter", range(31))
# def test_triton_persistent_matmul(iter):
#     Operator = load_opbench_by_name('gemm')
#     opbench = Operator(tb_args=parse_torchbench_args())
    
#     a, b, bias = opbench.get_example_inputs()

#     ans = opbench.triton_persistent_matmul(a, b, bias)()

#     assert ans is not None, "ans is None"
#     assert ans.device.type == 'cuda'


# enable this test after fixing the bug.
# RuntimeError: Triton Error [CUDA]: operation not supported

# @pytest.mark.parametrize("iter", range(31))
# def test_triton_tma_persistent_matmul(iter):
#     Operator = load_opbench_by_name('gemm')
#     opbench = Operator(tb_args=parse_torchbench_args())
    
#     a, b, bias = opbench.get_example_inputs()

#     ans = opbench.triton_tma_persistent_matmul(a, b, bias)()

#     assert ans is not None, "ans is None"
#     assert ans.device.type == 'cuda'


# enable this test after fixing the bug.
# RuntimeError: Triton Error [CUDA]: operation not supported

# @pytest.mark.parametrize("iter", range(31))
# def test_triton_tma_persistent_cached_matmul(iter):
#     Operator = load_opbench_by_name('gemm')
#     opbench = Operator(tb_args=parse_torchbench_args())
    
#     a, b, bias = opbench.get_example_inputs()

#     ans = opbench.triton_tma_persistent_cached_matmul(a, b, bias)()

#     assert ans is not None, "ans is None"
#     assert ans.device.type == 'cuda'

@pytest.mark.parametrize("iter", range(31))
def test_triton_ops_matmul(iter):
    Operator = load_opbench_by_name('gemm')
    opbench = Operator(tb_args=parse_torchbench_args())
    
    a, b, bias = opbench.get_example_inputs()

    ans = opbench.triton_ops_matmul(a, b, bias)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'


# enable this test after fixing the bug of "hammer missing"

# @pytest.mark.parametrize("iter", range(1))
# def test_hstu_triton_matmul(iter):
#     Operator = load_opbench_by_name('gemm')
#     opbench = Operator(tb_args=parse_torchbench_args())
    
#     a, b, bias = opbench.get_example_inputs()

#     ans = opbench.hstu_triton_matmul(a, b, bias)()

#     assert ans is not None, "ans is None"
#     assert ans.device.type == 'cuda'

@pytest.mark.parametrize("iter", range(31))
def test_pt2_triton_matmul(iter):
    Operator = load_opbench_by_name('gemm')
    opbench = Operator(tb_args=parse_torchbench_args())
    
    a, b, bias = opbench.get_example_inputs()

    ans = opbench.pt2_triton_matmul(a, b, bias)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'