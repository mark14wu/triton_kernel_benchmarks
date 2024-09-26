import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name
import torch

@pytest.mark.parametrize("iter", range(4))
def test_no_exp2(iter):
    Operator = load_opbench_by_name('template_attention')
    opbench = Operator(tb_args=parse_torchbench_args())

    arg0_1, arg1_1, arg2_1 = opbench.get_example_inputs()

    ans = opbench.test_no_exp2(arg0_1, arg1_1, arg2_1)()

    assert len(ans) == 1
    ans = ans[0]
    assert ans.device.type == 'cuda'

@pytest.mark.parametrize("iter", range(4))
def test_with_exp2(iter):
    Operator = load_opbench_by_name('template_attention')
    opbench = Operator(tb_args=parse_torchbench_args())

    arg0_2, arg1_2, arg2_2 = opbench.get_example_inputs()

    ans = opbench.test_with_exp2(arg0_2, arg1_2, arg2_2)()

    assert len(ans) == 1
    ans = ans[0]
    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'

if __name__ == "__main__":
    test_no_exp2(0)