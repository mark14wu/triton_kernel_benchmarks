import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name
import torch

@pytest.mark.parametrize("iter", range(8))
def test_triton_dropout(iter):
    Operator = load_opbench_by_name('low_mem_dropout')
    opbench = Operator(tb_args=parse_torchbench_args())
    
    p, x = opbench.get_example_inputs()

    ans = opbench.triton_dropout(p, x)()

