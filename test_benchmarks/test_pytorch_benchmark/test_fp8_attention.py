import pytest
import torch
from benchmark_utils import parse_torchbench_args
from torchbenchmark.operators import load_opbench_by_name

@pytest.mark.parametrize("iter", range(8))
def test_fp8_attention_triton_flash_v2(iter):
    Operator = load_opbench_by_name('fp8_attention')
    opbench = Operator(tb_args=parse_torchbench_args())

    q, k, v = opbench.get_example_inputs()

    assert q is not None
    assert k is not None
    assert v is not None
    assert q.device.type == 'cuda'
    assert k.device.type == 'cuda'
    assert v.device.type == 'cuda'
    # q, k, v will be converted to fp8 in the operator,
    # by 'triton_preprocess'
    assert q.dtype == torch.float16
    assert k.dtype == torch.float16
    assert v.dtype == torch.float16

    ans = opbench.triton_flash_v2(q, k, v)()
    assert ans is not None, "ans is None"
    assert ans.device.type == 'cuda'
    assert ans.dtype == torch.float8_e5m2

# @pytest.mark.parametrize("iter", range(6))
# def test_fp8_attention_colfax_fmha(iter):
#     Operator = load_opbench_by_name('fp8_attention')
#     opbench = Operator(tb_args=parse_torchbench_args())

#     q, k, v = opbench.get_example_inputs()

#     assert q is not None
#     assert k is not None
#     assert v is not None
#     assert q.device.type == 'cuda'
#     assert k.device.type == 'cuda'
#     assert v.device.type == 'cuda'
#     # q, k, v will be converted to fp8 in the operator,
#     # by 'colfax_preprocess'
#     assert q.dtype == torch.float16
#     assert k.dtype == torch.float16
#     assert v.dtype == torch.float16

#     ans = opbench.colfax_fmha(q, k, v)()
#     assert ans is not None, "ans is None"
#     assert ans.device.type == 'cuda'
#     assert ans.dtype == torch.float8_e4m3fn
