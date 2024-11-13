import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name
import torch

@pytest.mark.parametrize("iter", range(8))
def test_triton(iter):
    Operator = load_opbench_by_name('grouped_gemm')
    opbench = Operator(tb_args=parse_torchbench_args())
    
    group_A, group_B = opbench.get_example_inputs()

    ans_list = opbench.triton(group_A, group_B)()

    for ans in ans_list:
        assert ans.device.type == 'cuda'
