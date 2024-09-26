
# def test_softmax():
#     import torchbenchmark.operators.softmax.operator.Operator as softmax_operator_class
#     softmax_operator = softmax_operator_class()
#     x = torch.randn(10, 10, dtype=torch.float16, device='cuda')
#     y = softmax_operator.triton_softmax(x)
#     self.assertEqual(y.shape, x.shape)
#     self.assertTrue(torch.allclose(y.sum(dim=-1), torch.ones(y.size(0), dtype=torch.float16, atol=1e-6)))

import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name
import torch

@pytest.mark.parametrize("iter", range(98))
def test_triton_softmax(iter):
    Operator = load_opbench_by_name('softmax')
    opbench = Operator(tb_args=parse_torchbench_args())
    
    x = opbench.get_example_inputs()[0]

    ans = opbench.triton_softmax(x)()

    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'
