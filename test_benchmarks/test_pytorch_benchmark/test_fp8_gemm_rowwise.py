import torch
import pytest
from benchmark_utils import parse_torchbench_args, check_out_of_bounds
from torchbenchmark.operators import load_opbench_by_name

@pytest.mark.parametrize("iter", range(20))
def test_triton_fp8_gemm_no_fp8_fast_accum(iter):
    Operator = load_opbench_by_name('fp8_gemm_rowwise')
    opbench = Operator(
        tb_args=parse_torchbench_args(),
        extra_args=["--no_fp8_fast_accum"]
    )

    xq, wq, x_scale, w_scale = opbench.get_example_inputs()

    c = opbench._triton(xq, wq, x_scale, w_scale)()
    assert c.device.type == 'cuda'

    check_out_of_bounds()

@pytest.mark.parametrize("iter", range(20))
def test_triton_fp8_gemm_with_fp8_fast_accum(iter):
    Operator = load_opbench_by_name('fp8_gemm_rowwise')
    opbench = Operator(
        tb_args=parse_torchbench_args(),
        extra_args=[]
    )

    xq, wq, x_scale, w_scale = opbench.get_example_inputs()

    c = opbench._triton(xq, wq, x_scale, w_scale)()
    assert c.device.type == 'cuda'

    check_out_of_bounds()