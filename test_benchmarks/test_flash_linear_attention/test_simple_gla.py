# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla.ops.simple_gla.parallel import parallel_simple_gla


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("H", [1])
@pytest.mark.parametrize("T", [100, 512])
@pytest.mark.parametrize("D", [100, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_chunk(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    # [B, H, T, D]
    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_(True)
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_(True)
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_(True)
    g = torch.randn((B, H, T), dtype=dtype, device='cuda')
    h0 = torch.rand((B, H, D, D), dtype=torch.float32, device='cuda').requires_grad_(True)
    g = F.logsigmoid(g).requires_grad_(True)
    do = torch.randn_like(v)

    ref, ref_ht = fused_recurrent_simple_gla(q, k, v, g, initial_state=h0, output_final_state=True)
    d_ht = torch.randn_like(ref_ht)

    tri, tri_ht = chunk_simple_gla(q, k, v, g, initial_state=h0, output_final_state=True)
    ((tri * do).sum() + (tri_ht * d_ht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("H", [1])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("D", [32, 64, 100])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_parallel(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    q = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_(True)
    k = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_(True)
    v = torch.randn((B, H, T, D), dtype=dtype, device='cuda').requires_grad_(True)
    h0 = torch.zeros((B, H, D, D), dtype=torch.float32, device='cuda')
    g = F.logsigmoid(torch.randn((B, H, T), dtype=dtype, device='cuda'))
    g = (g / 16).requires_grad_(True)
    do = torch.randn_like(v)

    tri, _ = parallel_simple_gla(q, k, v, g)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None

