import pytest
import torch

from fla.ops.based import fused_chunk_based, parallel_based


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [8, 15])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_parallel_based(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    q = torch.randn((B, H, T, 16), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, 16), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)

    tri = parallel_based(q, k, v, use_norm=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [8, 15])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_chunk_based(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    q = torch.randn((B, H, T, 16), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, 16), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)

    tri = fused_chunk_based(q, k, v, use_norm=True)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
