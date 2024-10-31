import pytest
from benchmark_utils import parse_torchbench_args, check_out_of_bounds
from torchbenchmark.operators import load_opbench_by_name
import torch

@pytest.mark.parametrize("iter", range(10))
def test_no_welford(iter):
    Operator = load_opbench_by_name('welford')
    opbench = Operator(tb_args=parse_torchbench_args())
    
    p1, p2, p3 = opbench.get_example_inputs()

    opbench.test_no_welford(p1, p2, p3)()

    check_out_of_bounds()

@pytest.mark.parametrize("iter", range(10))
def test_welford(iter):
    Operator = load_opbench_by_name('welford')
    opbench = Operator(tb_args=parse_torchbench_args())

    p1, p2, p3 = opbench.get_example_inputs()

    opbench.test_welford(p1, p2, p3)()

    check_out_of_bounds()
