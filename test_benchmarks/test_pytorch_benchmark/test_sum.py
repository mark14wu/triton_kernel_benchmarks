import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name
import torch

# 20 tests are enough to cover the sum operator
@pytest.mark.parametrize("iter", range(20))
def test_triton_sum(iter):
    Operator = load_opbench_by_name('sum')
    opbench = Operator(tb_args=parse_torchbench_args())

    x = opbench.get_example_inputs()[0]

    ans = opbench.triton_sum(x)()
