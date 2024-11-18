import pytest
import torch
import torch.nn.functional as F

from fla.ops.hgrn import chunk_hgrn, fused_recurrent_hgrn


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [500, 1024])
@pytest.mark.parametrize("dtype", [torch.float])
def test_fused_recurrent(
    B: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    x = torch.randn((B, T, D), dtype=dtype, device='cuda')
    g = torch.randn((B, T, D), dtype=dtype, device='cuda')
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g))

    do = torch.randn_like(x)
    h0 = torch.randn_like(x[:, 0])

    tri, _ = fused_recurrent_hgrn(x, g, h0, output_final_state=True)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None


# TODO: support bfloat16
# @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [500, 1024])
@pytest.mark.parametrize("dtype", [torch.float])
def test_chunk(
    B: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    x = torch.randn((B, T, D), dtype=dtype, device='cuda')
    g = torch.randn((B, T, D), dtype=dtype, device='cuda')
    x, g = (1 - g.sigmoid()) * x, F.logsigmoid(g)
    x, g = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g))

    do = torch.randn_like(x)
    h0 = torch.randn_like(x[:, 0])

    tri, _ = chunk_hgrn(x, g, h0, output_final_state=True)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
