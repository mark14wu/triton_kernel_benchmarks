import torch
import pytest
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name


@pytest.mark.parametrize("iter", range(20))
def test_triton_fp8_gemm(iter):
    Operator = load_opbench_by_name('fp8_gemm')
    opbench = Operator(tb_args=parse_torchbench_args())

    a, b = opbench.get_example_inputs()
    assert a.device.type == 'cuda'
    assert b.device.type == 'cuda'
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn

    c = opbench.triton_fp8_gemm(a, b)()
    assert c.device.type == 'cuda'
    assert c.shape == (a.shape[0], b.shape[1])

@pytest.mark.parametrize("iter", range(20))
def test_triton_persistent_fp8_gemm(iter):
    Operator = load_opbench_by_name('fp8_gemm')
    opbench = Operator(tb_args=parse_torchbench_args())

    a, b = opbench.get_example_inputs()
    assert a.device.type == 'cuda'
    assert b.device.type == 'cuda'
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn

    c = opbench.triton_persistent_fp8_gemm(a, b)()
    assert c.device.type == 'cuda'
    assert c.shape == (a.shape[0], b.shape[1])
    assert c.dtype == torch.float8_e4m3fn

@pytest.mark.parametrize("iter", range(20))
def test_triton_tma_persistent_fp8_gemm(iter):
    Operator = load_opbench_by_name('fp8_gemm')
    opbench = Operator(tb_args=parse_torchbench_args())

    a, b = opbench.get_example_inputs()
    assert a.device.type == 'cuda'
    assert b.device.type == 'cuda'
    assert a.dtype == torch.float8_e4m3fn

    c = opbench.triton_tma_persistent_fp8_gemm(a, b)()
    assert c.device.type == 'cuda'
    assert c.shape == (a.shape[0], b.shape[1])
    assert c.dtype == torch.float8_e4m3fn
