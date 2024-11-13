import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name
import torch

@pytest.mark.parametrize("iter", range(42))
def test_triton_addmm(iter):
    Operator = load_opbench_by_name('addmm')
    opbench = Operator(tb_args=parse_torchbench_args())

    a, mat1, mat2 = opbench.get_example_inputs()

    ans = opbench.triton_addmm(a, mat1, mat2)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'
