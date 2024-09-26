import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name
import torch

@pytest.mark.parametrize("iter", range(30))
def test_triton_layer_norm(iter):
    Operator = load_opbench_by_name('layer_norm')
    opbench = Operator(tb_args=parse_torchbench_args())
    
    x, w_shape, weight, bias, eps = opbench.get_example_inputs()

    ans = opbench.triton_layer_norm(x, w_shape, weight, bias, eps)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'
