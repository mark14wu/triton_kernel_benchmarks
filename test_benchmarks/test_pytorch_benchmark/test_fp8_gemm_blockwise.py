import torch
import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name

@pytest.mark.parametrize("iter", range(20))
def test_triton_fp8_gemm(iter):
    Operator = load_opbench_by_name('fp8_gemm_blockwise')
    opbench = Operator(tb_args=parse_torchbench_args())

    xq, wq, x_scale, w_scale = opbench.get_example_inputs()

    c = opbench._triton(xq, wq, x_scale, w_scale)()
    assert c.device.type == 'cuda'
