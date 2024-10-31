import pytest
from benchmark_utils import parse_torchbench_args, check_out_of_bounds
from torchbenchmark.operators import load_opbench_by_name
import torch

@pytest.mark.parametrize("iter", range(16))
def test_triton_add(iter):
    Operator = load_opbench_by_name('vector_add')
    opbench = Operator(tb_args=parse_torchbench_args())
    
    x, y = opbench.get_example_inputs()

    ans = opbench.triton_add(x, y)()
    check_out_of_bounds()