from cairo import Operator
import pytest
from benchmark_utils import parse_torchbench_args, check_out_of_bounds
from torchbenchmark.operators import load_opbench_by_name
import torch

@pytest.mark.parametrize("iter", range(20))
def test_rms_norm_fused(iter):
    Operator = load_opbench_by_name('fp8_fused_quant_gemm_rowwise')
    opbench = Operator(tb_args=parse_torchbench_args())

    x1, x2, wq, w_scale, wd = opbench.get_example_inputs()

    ans = opbench.rms_norm_fused(x1, x2, wq, w_scale, wd)

    check_out_of_bounds()

@pytest.mark.parametrize("iter", range(20))
def test_rms_norm_quant(iter):
    Operator = load_opbench_by_name('fp8_fused_quant_gemm_rowwise')
    opbench = Operator(tb_args=parse_torchbench_args())

    x1, x2, wq, w_scale, wd = opbench.get_example_inputs()

    ans = opbench.rms_norm_quant(x1, x2, wq, w_scale, wd)

    check_out_of_bounds()

@pytest.mark.parametrize("iter", range(20))
def test_silu_mul_fused(iter):
    Operator = load_opbench_by_name('fp8_fused_quant_gemm_rowwise')
    opbench = Operator(tb_args=parse_torchbench_args())

    x1, x2, wq, w_scale, wd = opbench.get_example_inputs()

    ans = opbench.silu_mul_fused(x1, x2, wq, w_scale, wd)

    check_out_of_bounds()

@pytest.mark.parametrize("iter", range(20))
def test_silu_mul_quant(iter):
    Operator = load_opbench_by_name('fp8_fused_quant_gemm_rowwise')
    opbench = Operator(tb_args=parse_torchbench_args())

    x1, x2, wq, w_scale, wd = opbench.get_example_inputs()

    ans = opbench.silu_mul_quant(x1, x2, wq, w_scale, wd)
    
    check_out_of_bounds()